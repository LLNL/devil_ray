#include <dray/filters/mesh_boundary.hpp>

#include <dray/dispatcher.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/elem_utils.hpp>
#include <dray/Element/elem_ops.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{
namespace detail
{

template<class MElemT>
DataSet
boundary_execute(Mesh<MElemT> &mesh, Array<Vec<int32, 2>> &elid_faceid_state)
{
  constexpr ElemType etype = MElemT::get_etype();

  using OutMeshElement =
    Element<MElemT::get_dim()-1, 3, MElemT::get_etype (), MElemT::get_P ()>;

  Mesh<MElemT> orig_mesh = mesh;
  const int32 mesh_poly_order = orig_mesh.get_poly_order();

  //
  // Step 1: Extract the boundary mesh: Matt's external_faces() algorithm.
  //

  // Identify unique/external faces.
  Array<Vec<int32,4>> face_corner_ids = detail::extract_faces(orig_mesh);
  Array<int32> orig_face_idx = detail::sort_faces(face_corner_ids);
  detail::unique_faces(face_corner_ids, orig_face_idx);
  elid_faceid_state = detail::reconstruct<etype>(orig_face_idx);

  // Copy the dofs for each face.
  // The template argument '3u' means 3 components (embedded in 3D).
  GridFunction<3u> mesh_data_2d
      = detail::extract_face_dofs(Shape<3, etype>{},
                                  orig_mesh.get_dof_data(),
                                  mesh_poly_order,
                                  elid_faceid_state);

  // Wrap the mesh data inside a mesh and dataset.
  Mesh<OutMeshElement> boundary_mesh(mesh_data_2d, mesh_poly_order);

  DataSet out_data_set(std::make_shared<DerivedTopology<OutMeshElement>>(boundary_mesh));

  return out_data_set;
}

template <typename FElemT>
std::shared_ptr<FieldBase>
boundary_field_execute(Field<FElemT> &in_field,
                       const Array<Vec<int32, 2>> &elid_faceid_state)
{
  //
  // Step 2: For each field, add boundary field to the boundary_dataset.
  //
  // We already know what kind of elements we have
  constexpr int32 in_dim = FElemT::get_dim();
  constexpr int32 ncomp = FElemT::get_ncomp();
  constexpr ElemType etype = FElemT::get_etype();
  constexpr int32 P = FElemT::get_P();

  const std::string fname = in_field.name();
  const int32 field_poly_order = in_field.order();

  GridFunction<FElemT::get_ncomp()> out_data
      = detail::extract_face_dofs(Shape<3, etype>{},
                                  in_field.get_dof_data(),
                                  field_poly_order,
                                  elid_faceid_state);

  // Reduce dimension, keep everything else the same as input.
  using OutFElemT = Element<in_dim-1, ncomp, etype, P>;

  std::shared_ptr<Field<OutFElemT>> out_field
    = std::make_shared<Field<OutFElemT>>(out_data, field_poly_order);
  out_field->name(fname);

  return out_field;
}


struct BoundaryFieldFunctor
{
  const Array<Vec<int32, 2>> m_elid_faceid;

  std::shared_ptr<FieldBase> m_output;

  BoundaryFieldFunctor(const Array<Vec<int32, 2>> elid_faceid)
    : m_elid_faceid{elid_faceid}
  { }

  template <typename TopoType, typename FieldType>
  void operator()(TopoType &, FieldType &in_field)
  {
    m_output = boundary_field_execute(in_field, m_elid_faceid);
  }
};


struct BoundaryFunctor
{
  DataSet m_input;
  Array<Vec<int32, 2>> m_elid_faceid;
  DataSet m_output;

  BoundaryFunctor(DataSet &input)
    : m_input(input)
  { }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    DRAY_LOG_OPEN("mesh_boundary");
    m_output = boundary_execute(topo.mesh(), m_elid_faceid);

    // TODO: Currently we only support extracting scalar fields.
    // (We could support vector fields with another dispatch attempt).
    // We should support vector fields, but i don't know what we
    // would do with a 2d/1d vector field in 3d space

    const int32 num_fields = m_input.number_of_fields();
    for (int32 field_idx = 0; field_idx < num_fields; field_idx++)
    {
      FieldBase * field_base = m_input.field(field_idx);
      BoundaryFieldFunctor bff(m_elid_faceid);
      try
      {
        dispatch_scalar_field(field_base, &topo, bff);
        m_output.add_field(bff.m_output);
      }
      catch (const DRayError &dispatch_excpt)
      {
        std::cerr << "Boundary: Field '" << field_base->name() << "' not supported. Skipping. "
                  << "Reason: " << dispatch_excpt.GetMessage() << "\n";
      }
    }
    DRAY_LOG_CLOSE();
  }
};


}//namespace detail

Collection
MeshBoundary::execute(Collection &collection)
{
  Collection res;
  for(int32 i = 0; i < collection.size(); ++i)
  {
    DataSet data_set = collection.domain(i);
    if(data_set.topology()->dims() == 3)
    {
      detail::BoundaryFunctor func(data_set);
      dispatch_3d(data_set.topology(), func);
      res.add_domain(func.m_output);
    }
    else
    {
      // just pass it through
      res.add_domain(data_set);
    }
  }
  return res;
}


}//namespace dray

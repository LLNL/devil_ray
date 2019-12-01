#include <dray/filters/mesh_boundary.hpp>

#include <dray/data_set.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{
namespace detail
{

template <int32 ndof>
GridFunction<ndof> extract_face_dofs(const GridFunction<ndof> &orig_data_3d,
                                         const int32 poly_order,
                                         const Array<Vec<int32, 2>> &elid_faceid)
{
  // copy_face_dof_subset
  // - If true, make a new array filled with just the surface geometry
  //   The dof ids for the surface mesh will no longer correspond to
  //   their ids in the volume mesh.
  // - If false, the old geometry array will be re-used, so dof ids
  //   will match the volume mesh.
  const bool copy_face_dof_subset = false;

  GridFunction<ndof> new_data_2d;

  if (copy_face_dof_subset)    // New geometry array with subset of dofs.
  {
    //TODO
  }
  else                         // Re-use the old geometry, make new topology.
  {
    // Make sure to initialize all 5 members. TODO a new constructor?
    new_data_2d.m_el_dofs = (poly_order+1)*(poly_order+1);
    new_data_2d.m_size_el = elid_faceid.size();
    new_data_2d.m_size_ctrl = orig_data_3d.m_size_ctrl;
    new_data_2d.m_values = orig_data_3d.m_values;

    new_data_2d.m_ctrl_idx.resize((poly_order+1)*(poly_order+1) * elid_faceid.size());

    const Vec<int32,2> * elid_faceid_ptr = elid_faceid.get_device_ptr_const();
    const int32 * orig_dof_idx_ptr       = orig_data_3d.m_ctrl_idx.get_device_ptr_const();
    int32 * new_dof_idx_ptr              = new_data_2d.m_ctrl_idx.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, new_data_2d.m_size_el), [=] DRAY_LAMBDA (int32 face_idx)
    {
      //
      // Convention for face ids and lexicographic ordering of dofs:
      //
      // =========   =========   =========    =========   =========   =========
      // faceid 0:   faceid 1:   faceid 2:    faceid 3:   faceid 4:   faceid 5:
      // z^          z^          y^           z^          z^          y^
      //  |(4) (6)    |(4) (5)    |(2) (3)     |(5) (7)    |(6) (7)    |(6) (7)
      //  |           |           |            |           |           |
      //  |(0) (2)    |(0) (1)    |(0) (1)     |(1) (3)    |(2) (3)    |(4) (5)
      //  ------->    ------->    ------->     ------->    ------->    ------->
      // (x=0)   y   (y=0)   x   (z=0)   x    (X=1)   y   (Y=1)   x   (Z=1)   x
      //

      const int32 eldofs0 = 1;
      const int32 eldofs1 = eldofs0 * (poly_order+1);
      const int32 eldofs2 = eldofs1 * (poly_order+1);
      const int32 eldofs3 = eldofs2 * (poly_order+1);
      const int32 axis_strides[3] = {eldofs0, eldofs1, eldofs2};

      const int32 faceid      = elid_faceid_ptr[face_idx][1];
      const int32 orig_offset = elid_faceid_ptr[face_idx][0] * eldofs3;
      const int32 new_offset  = face_idx * eldofs2;

      const int32 face_axis = (faceid == 0 || faceid == 3 ? 0    // Conditional
                             : faceid == 1 || faceid == 4 ? 1    // instead of
                                                          : 2);  //    % /

      const int32 face_start = (faceid < 3 ? 0 : (eldofs1 - 1) * axis_strides[face_axis]);
      const int32 major_stride = axis_strides[(face_axis == 2 ? 1 : 2)];
      const int32 minor_stride = axis_strides[(face_axis == 0 ? 1 : 0)];

      for (int32 ii = 0; ii < eldofs1; ii++)
        for (int32 jj = 0; jj < eldofs1; jj++)
          new_dof_idx_ptr[new_offset + eldofs1*ii + jj] =
              orig_dof_idx_ptr[orig_offset + face_start + major_stride*ii + minor_stride*jj];
    });
  }

  return new_data_2d;
}
}//namespace detail

template<class ElemT>
DataSet<NDElem<ElemT, 2>>
MeshBoundary::execute(DataSet<ElemT> &data_set)
{
  DRAY_LOG_OPEN("mesh_boundary");
  using Elem3D = ElemT;
  using Elem2D = NDElem<ElemT, 2>;

  Mesh<ElemT> orig_mesh = data_set.get_mesh();
  const int32 mesh_poly_order = orig_mesh.get_poly_order();

  //
  // Step 1: Extract the boundary mesh: Matt's external_faces() algorithm.
  //

  // Identify unique/external faces.
  Array<Vec<int32,4>> face_corner_ids = detail::extract_faces(orig_mesh);
  Array<int32> orig_face_idx = detail::sort_faces(face_corner_ids);
  detail::unique_faces(face_corner_ids, orig_face_idx);
  Array<Vec<int32, 2>> elid_faceid = detail::reconstruct(orig_face_idx);

  // Copy the dofs for each face.
  // The template argument '3u' means 3 components (embedded in 3D).
  GridFunction<3u> mesh_data_2d
      = detail::extract_face_dofs(orig_mesh.get_dof_data(),
                                  mesh_poly_order,
                                  elid_faceid);

  // Wrap the mesh data inside a mesh and dataset.
  Mesh<Elem2D> boundary_mesh(mesh_data_2d, mesh_poly_order);
  DataSet<Elem2D> boundary_dataset(boundary_mesh);

  //
  // Step 2: For each field, add boundary field to the boundary_dataset.
  //
  const int32 num_fields = data_set.number_of_fields();
  for (int32 field_idx = 0; field_idx < num_fields; field_idx++)
  {
    Field<FieldOn<ElemT, 1u>> orig_field = data_set.get_field(field_idx);
    const int32 field_poly_order = orig_field.order();

    // Extract surface-only dofs.
    // The template argument '1u' means scalar field.
    GridFunction<1u> mesh_data_2d
        = detail::extract_face_dofs(orig_field.get_dof_data(),
                                field_poly_order,
                                elid_faceid);

    // Wrap the new 2d field data inside a field and add to the dataset.
    const std::string field_name = data_set.get_field_name(field_idx);
    Field<FieldOn<Elem2D, 1u>> field_2d(mesh_data_2d, field_poly_order);
    boundary_dataset.add_field(field_2d, field_name);
  }

  DRAY_LOG_CLOSE();
  return boundary_dataset;
}



  // Explicit instantiations.
  //

  template
    DataSet<MeshElem<2u, ElemType::Quad, Order::General>>
    MeshBoundary::execute<MeshElem<3u, ElemType::Quad, Order::General>>(
        DataSet<MeshElem<3u, ElemType::Quad, Order::General>> &data_set);

  // <float32, Tri>
  /// template
  ///   DataSet<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>
  ///   MeshBoundary::execute<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>(
  ///       DataSet<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>> &data_set);

  // <float64, Tri>
  /// template
  ///   DataSet<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>
  ///   MeshBoundary::execute<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>(
  ///       DataSet<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>> &data_set);

}//namespace dray

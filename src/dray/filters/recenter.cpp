#include <dray/filters/recenter.hpp>

#include <dray/dispatcher.hpp>
#include <dray/Element/elem_utils.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{

namespace detail
{

template<typename MeshElem>
//Array<int32>
void
conn_counts(Array<int32> &corners, Mesh<MeshElem> &mesh)
{
  const int corner_size = corners.size();
  const int32 mesh_dof_size = mesh.get_dof_data().m_ctrl_idx.size();
  Array<int32> sparse_counts;
  sparse_counts.resize(mesh_dof_size);
  array_memset(sparse_counts, 0);
  int32 * counts_ptr = sparse_counts.get_device_ptr();
  int32 * corners_ptr = corners.get_device_ptr();

  RAJA::forall<for_policy>
    (RAJA::RangeSegment (0, corner_size), [=] DRAY_LAMBDA (int32 i)
  {
    int32 index = corners_ptr[i];
    int32 old = RAJA::atomicAdd<atomic_policy> (&(counts_ptr[index]), 1);
  });
  DRAY_ERROR_CHECK();

  RAJA::ReduceSum<reduce_policy, int32> total_counts (0);

  RAJA::forall<for_policy>
    (RAJA::RangeSegment (0, mesh_dof_size), [=] DRAY_LAMBDA (int32 i)
  {
    total_counts += counts_ptr[i];
  });
  DRAY_ERROR_CHECK();
  std::cout<<"total count "<<total_counts.get()<<"\n";



}

template<typename MeshElem,
         typename FieldElem>
//DataSet
void
recenter_execute(Mesh<MeshElem> &mesh,
                 Field<FieldElem> &field)
{
  DRAY_LOG_OPEN("recenter");

  // im afraid of lambda capture

  GridFunction<3u> mesh_gf = mesh.get_dof_data();
  GridFunction<FieldElem::get_ncomp()> field_gf = field.get_dof_data();

  int32 corners_per;
  Array<int32> corners = detail::extract_corners(mesh, corners_per);
  conn_counts(corners,mesh);


  //Array<Vec<int32, 4>> faces = extract_faces(mesh);

  //// shallow copy everything
  //GridFunction<3u> output_gf = input_gf;
  //// deep copy values
  //Array<Vec<Float, 3>> points;
  //array_copy (points, input_gf.m_values);

  //Vec<Float,3> *points_ptr = points.get_device_ptr();
  //const int size = points.size();

  //RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  //{
  //  Vec<Float,3> dof = points_ptr[i];
  //  Vec<Float,3> dir = dof - lpoint;
  //  Float dist = dot(dir, lnormal);
  //  dof = dof - Float(2.f) * dist * lnormal;
  //  points_ptr[i] = dof;
  //});

  //// replace the input values
  //output_gf.m_values = points;

  //Mesh<MeshElem> out_mesh(output_gf, mesh.get_poly_order());
  //std::shared_ptr<DerivedTopology<MeshElem>> topo
  //  = std::make_shared<DerivedTopology<MeshElem>>(out_mesh);
  //DataSet dataset(topo);

  DRAY_LOG_CLOSE();
  //return dataset;
}

struct RecenterFunctor
{
  DataSet m_res;
  RecenterFunctor()
  {
  }

  template<typename TopologyType, typename FieldType>
  void operator()(TopologyType &topo, FieldType &field)
  {
    //m_res = detail::recenter_execute(topo.mesh(), field);
    detail::recenter_execute(topo.mesh(), field);
  }
};

}//namespace detail

Recenter::Recenter()
{
}

Collection
Recenter::execute(Collection &collection)
{
  if(m_field_name == "")
  {
    DRAY_ERROR("Recenter: field never set");
  }
  Collection res;
  for(int32 i = 0; i < collection.size(); ++i)
  {
    DataSet data_set = collection.domain(i);
    FieldBase *field = data_set.field(m_field_name);
    TopologyBase *topo = data_set.topology();

    detail::RecenterFunctor func;
    dispatch_3d(topo, field,func);

    // pass through all in the input fields
    //const int num_fields = data_set.number_of_fields();
    //for(int i = 0; i < num_fields; ++i)
    //{
    //  func.m_res.add_field(data_set.field_shared(i));
    //}
    //res.add_domain(func.m_res);
  }
  return res;
}

void
Recenter::field(const std::string field_name)
{
  m_field_name = field_name;
}

}//namespace dray

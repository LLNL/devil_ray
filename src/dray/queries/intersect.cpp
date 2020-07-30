#include <dray/queries/intersect.hpp>

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
void
intersect_execute(Mesh<MeshElem> &mesh,
                  Array<Vec<Float,3>> &ips,
                  Array<Vec<Float,3>> &dirs)
{
  DRAY_LOG_OPEN("reflect");

 // // im afraid of lambda capture
 // const Vec<Float,3> lpoint = point;
 // const Vec<Float,3> lnormal = normal;

 // GridFunction<3u> input_gf = mesh.get_dof_data();
 // // shallow copy everything
 // GridFunction<3u> output_gf = input_gf;
 // // deep copy values
 // Array<Vec<Float, 3>> points;
 // array_copy (points, input_gf.m_values);

 // Vec<Float,3> *points_ptr = points.get_device_ptr();
 // const int size = points.size();

 // RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
 // {
 //   Vec<Float,3> dof = points_ptr[i];
 //   Vec<Float,3> dir = dof - lpoint;
 //   Float dist = dot(dir, lnormal);
 //   dof = dof - Float(2.f) * dist * lnormal;
 //   points_ptr[i] = dof;
 // });

 // // replace the input values
 // output_gf.m_values = points;

 // Mesh<MeshElem> out_mesh(output_gf, mesh.get_poly_order());
 // std::shared_ptr<DerivedTopology<MeshElem>> topo
 //   = std::make_shared<DerivedTopology<MeshElem>>(out_mesh);
 // DataSet dataset(topo);

 // DRAY_LOG_CLOSE();
//  return dataset;
}

struct IntersectFunctor
{
  Array<Vec<Float,3>> m_ips;
  Array<Vec<Float,3>> m_dirs;
  IntersectFunctor(Array<Vec<Float,3>> &ips,
                   Array<Vec<Float,3>> &dirs)
    : m_ips(ips)
    , m_dirs(dirs)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    detail::intersect_execute(topo.mesh(), m_ips, m_dirs);
  }
};

}//namespace detail

Intersect::Intersect()
{
}

void
Intersect::execute(Collection &collection,
                   const std::vector<Vec<float64,3>> &directions,
                   const std::vector<Vec<float64,3>> &ips,
                   Array<int32> &face_ids,
                   Array<Vec<Float,3>> &res_ips)
{

}

}//namespace dray

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

int32 num_faces(ShapeHex &)
{
  return 6;
}

int32 num_face_dofs(ShapeHex &, const int p)
{
  return (p + 1) * (p + 1);
}

int32 num_faces(ShapeTet &)
{
  return 4;
}

int32 num_face_dofs(ShapeTet &, const int p)
{
  return (p + 1) / 1 * (p + 2) / 2;
}


template<typename MeshElem>
void
intersect_execute(Mesh<MeshElem> &mesh,
                  Array<Vec<Float,3>> &ips,
                  Array<Vec<Float,3>> &dirs,
                  Array<int32> &face_ids,
                  Array<Vec<Float,3>> &res_ips)
{
  DRAY_LOG_OPEN("intesect");
  const int32 num_ips = ips.size();
  const int32 num_dirs = dirs.size();
  const int32 num_elems = mesh.get_num_elem();
  std::cout<<"Rays per elem "<<num_ips * num_dirs<<"\n";
  std::cout<<"Elements "<<num_elems<<"\n";

  constexpr ElemType etype = MeshElem::get_etype();
  const int32 poly_order = mesh.get_poly_order();

  Shape<3, etype> shape;
  const int32 face_count = num_faces(shape);
  const int32 face_dofs = num_face_dofs(shape, poly_order);

  const GridFunction<3> &mesh_dofs = mesh.get_dof_data();

  // we need scratch space to put faces into
  // 1 face per element
  GridFunction<3> scratch_gf;
  scratch_gf.resize_counting(num_elems, face_dofs);

  Array<AABB<3>> face_aabbs;
  face_aabbs.resize(face_count * num_elems);

  AABB<3> *face_aabbs_ptr = face_aabbs.get_device_ptr();

  DeviceMesh<MeshElem> device_mesh(mesh);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 i)
  {
    MeshElem elem = device_mesh.get_elem(i);
  });

 // // replace the input values
 // output_gf.m_values = points;

 // Mesh<MeshElem> out_mesh(output_gf, mesh.get_poly_order());
 // std::shared_ptr<DerivedTopology<MeshElem>> topo
 //   = std::make_shared<DerivedTopology<MeshElem>>(out_mesh);
 // DataSet dataset(topo);

  DRAY_LOG_CLOSE();
}

struct IntersectFunctor
{
  Array<Vec<Float,3>> m_ips;
  Array<Vec<Float,3>> m_dirs;
  Array<int32> face_ids;
  Array<Vec<Float,3>> res_ips;

  IntersectFunctor(Array<Vec<Float,3>> &ips,
                   Array<Vec<Float,3>> &dirs)
    : m_ips(ips)
    , m_dirs(dirs)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    detail::intersect_execute(topo.mesh(),
                              m_ips,
                              m_dirs,
                              face_ids,
                              res_ips);
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
  Array<Vec<Float,3>> a_dirs;
  Array<Vec<Float,3>> a_ips;

  a_dirs.resize(directions.size());
  a_ips.resize(ips.size());

  Vec<Float,3> *dir_ptr = a_dirs.get_host_ptr();
  for(int i = 0; i < directions.size(); ++i)
  {
    dir_ptr[i] = directions[i];
  }

  Vec<Float,3> *ip_ptr = a_ips.get_host_ptr();
  for(int i = 0; i < ips.size(); ++i)
  {
    ip_ptr[i] = ips[i];
  }

  const int num_domains = collection.size();
  for(int i = 0; i < num_domains; ++i)
  {
    DataSet dataset = collection.domain(i);
    detail::IntersectFunctor func(a_dirs, a_ips);
    dispatch_3d(dataset.topology(), func);
  }

}

}//namespace dray

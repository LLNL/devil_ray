#include <dray/queries/intersect.hpp>

#include <dray/dispatcher.hpp>
#include <dray/face_intersection.hpp>
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

template<int32 Dims, ElemType etype, int32 P>
Mesh<Element<Dims-1,3,etype,P>>
face_mesh(Mesh<Element<Dims,3,etype,P>> &mesh)
{
  using MeshElem = Element<Dims,3,etype,P>;
  const int32 num_elems = mesh.get_num_elem();
  Shape<3, etype> shape;
  const int32 face_count = num_faces(shape);
  const int32 face_dofs = num_face_dofs(shape, P);
  const GridFunction<3> &mesh_dofs = mesh.get_dof_data();

  // create element face id pairs so we can
  // extract all the faces into a submesh
  Array<Vec<int32,2>> elem_face_ids;
  elem_face_ids.resize(face_count * num_elems);
  Vec<int32,2> *elem_face_ids_ptr = elem_face_ids.get_device_ptr();

  const int32 mesh_poly_order = mesh.get_poly_order();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 i)
  {
    const int offset = num_elems * face_count * i;
    Vec<int32,2> pair;
    pair[0] = i;
    for(int f = 0; f < face_count; ++f)
    {
      pair[1] = f;
      elem_face_ids_ptr[offset + f] = pair;
    }
  });

  GridFunction<3> face_gf = detail::extract_face_dofs(shape,
                                                      mesh_dofs,
                                                      mesh_poly_order,
                                                      elem_face_ids);

  constexpr int32 in_dim = MeshElem::get_dim();

  using FaceElem = Element<Dims-1, 3, etype, P>;
  Mesh<FaceElem> faces(face_gf, mesh_poly_order);
  return faces;
}

DRAY_EXEC
Float ray_aabb(const Ray &ray, const AABB<3> &aabb)
{
  const Vec<Float, 3> ray_dir = ray.m_dir;
  const Vec<Float, 3> ray_orig = ray.m_orig;

  float32 dirx = static_cast<float32> (ray_dir[0]);
  float32 diry = static_cast<float32> (ray_dir[1]);
  float32 dirz = static_cast<float32> (ray_dir[2]);
  float32 origx = static_cast<float32> (ray_orig[0]);
  float32 origy = static_cast<float32> (ray_orig[1]);
  float32 origz = static_cast<float32> (ray_orig[2]);

  const float32 inv_dirx = rcp_safe (dirx);
  const float32 inv_diry = rcp_safe (diry);
  const float32 inv_dirz = rcp_safe (dirz);

  const float32 odirx = origx * inv_dirx;
  const float32 odiry = origy * inv_diry;
  const float32 odirz = origz * inv_dirz;

  const float32 xmin = aabb.m_ranges[0].min () * inv_dirx - odirx;
  const float32 ymin = aabb.m_ranges[1].min () * inv_diry - odiry;
  const float32 zmin = aabb.m_ranges[2].min () * inv_dirz - odirz;
  const float32 xmax = aabb.m_ranges[0].max () * inv_dirx - odirx;
  const float32 ymax = aabb.m_ranges[1].max () * inv_diry - odiry;
  const float32 zmax = aabb.m_ranges[2].max () * inv_dirz - odirz;

  const float32 min_int = ray.m_near;
  float32 min_dist =
    max (max (max (min(ymin, ymax), min(xmin, xmax)), min(zmin, zmax)), min_int);
  float32 max_dist = min(min(max(ymin, ymax), max(xmin, xmax)), max(zmin, zmax));
  max_dist = min(max_dist, float32(ray.m_far));

  Float res = -1.f;
  if(max_dist > min_dist)
  {
    res = min_dist;
  }
  return res;
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

  constexpr int32 in_dim = MeshElem::get_dim();
  constexpr int32 P = MeshElem::get_P();

  using FaceElem = Element<in_dim-1, 3, etype, P>;

  Shape<3, etype> shape;
  const int32 face_count = num_faces(shape);

  // Note: there is a side effect that a BVH exists over faces
  // along with subdivisions and aabbs. Could use this to create
  // mapping between subdivisions and elements to get some guesses.
  // There are duplicate faces present
  Mesh<FaceElem> faces_mesh = face_mesh(mesh);

  // face elements are in the order

  // =========   =========   =========    =========   =========   =========
  // faceid 0:   faceid 1:   faceid 2:    faceid 3:   faceid 4:   faceid 5:
  // z^          z^          y^           z^          z^          y^
  //  |(4) (6)    |(4) (5)    |(2) (3)     |(5) (7)    |(6) (7)    |(6) (7)
  //  |           |           |            |           |           |
  //  |(0) (2)    |(0) (1)    |(0) (1)     |(1) (3)    |(2) (3)    |(4) (5)
  //  ------->    ------->    ------->     ------->    ------->    ------->
  // (x=0)   y   (y=0)   x   (z=0)   x    (X=1)   y   (Y=1)   x   (Z=1)   x

  DeviceMesh<FaceElem> d_face_mesh(faces_mesh);
  DeviceMesh<MeshElem> d_mesh(mesh);

  const int32 dispatch_size = num_ips * num_dirs * num_elems;
  std::cout<<"Dispatch size "<<dispatch_size<<"\n";
  Vec<Float,3> *ips_ptr = ips.get_device_ptr();
  Vec<Float,3> *dirs_ptr = dirs.get_device_ptr();

  // we keeps stats
  Array<stats::Stats> mstats;
  mstats.resize(dispatch_size);
  stats::Stats *mstats_ptr = mstats.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, dispatch_size),
      [=] DRAY_LAMBDA (int32 i)
  {
    // todo figure out the best ordering
    // we are treating this as a 3d indexing where
    // ips is varying the fastest, then dirs, then elements
    const int32 ip_id = i % num_ips;
    const int32 dir_id = (i / num_ips) % num_dirs;
    const int32 el_id = i / (num_ips * num_dirs);

    const Vec<Float,3> ip = ips_ptr[ip_id];
    const Vec<Float,3> dir = dirs_ptr[dir_id];

    MeshElem cell = d_mesh.get_elem(el_id);

    Ray ray;
    ray.m_orig = cell.eval(ip);
    ray.m_dir = dir;
    ray.m_near = 0.f;
    ray.m_far = infinity<Float>();

    const int32 face_offset = el_id * face_count;

    Float distances[face_count];

    // see which faces we hit. This may or may not be
    // a good way to rule faces out. In the case where
    // the faces are the opposite of axis aligned, a face
    // aabb could contain the origin, even when the ray
    // is pointinig in the other direction. In then
    // case where perfectly axis aligned, this is a
    // perfect test.
    for(int32 f = 0; f < face_count; ++f)
    {
      FaceElem face = d_face_mesh.get_elem(face_offset + f);
      AABB<3> face_bounds;
      face.get_bounds(face_bounds);
      distances[f] = ray_aabb(ray, face_bounds);
      if(distances[f] >= 0.f) std::cout<<"Face "<<f<<" d "<<distances[f]<<"\n";
    }

    Vec<Float,2> ref = {0.5f, 0.5f};
    bool hit = false;

    stats::Stats mstat;
    mstat.construct();

    for(int32 f = 0; f < face_count; ++f)
    {
      FaceElem face = d_face_mesh.get_elem(face_offset + f);
      if(distances[f] >= 0.f)
      {
        bool use_guess = false;
        hit = Intersector_RayFace<FaceElem>::intersect_local (mstat,
                                                              face,
                                                              ray,
                                                              ref,
                                                              distances[f],
                                                              use_guess);
        if(hit)
        {
          std::cout<<"Hit "<<f<<" distance "<<distances[f]<<"\n";
          mstat.found();
          //break;
        }
      }
    }

    mstats_ptr[i] = mstat;

  });

  //stats::StatStore::add_ray_stats(rays, mstats);
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
    dir_ptr[i].normalize();
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

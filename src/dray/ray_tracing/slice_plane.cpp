#include <dray/ray_tracing/slice_plane.hpp>
#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <assert.h>

namespace dray
{
namespace ray_tracing
{

namespace detail
{

Array<RayHit>
get_hits(const Array<Ray> &rays,
         const Array<Location> &locations,
         const Array<Vec<Float,3>> &points)
{
  Array<RayHit> hits;
  hits.resize(rays.size());

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const Location *loc_ptr = locations.get_device_ptr_const();
  const Vec<Float,3> *points_ptr = points.get_device_ptr_const();
  RayHit *hit_ptr = hits.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, rays.size()), [=] DRAY_LAMBDA (int32 i)
  {
    RayHit hit;
    const Location loc = loc_ptr[i];
    const Ray ray = ray_ptr[i];
    const Vec<Float,3> point = points_ptr[i];
    hit.m_hit_idx = loc.m_cell_id;
    hit.m_ref_pt  = loc.m_ref_pt;

    if(hit.m_hit_idx > -1)
    {
      hit.m_dist = (point - ray.m_orig).magnitude();
    }

    hit_ptr[i] = hit;

  });
  return hits;
}

Array<Vec<Float,3>>
calc_sample_points(Array<Ray> &rays,
                   const Vec<float32,3> &point,
                   const Vec<float32,3> &normal)
{
  const int32 size = rays.size();

  Array<Vec<Float,3>> points;
  points.resize(size);

  Vec<Float,3> t_normal;
  t_normal[0] = normal[0];
  t_normal[1] = normal[1];
  t_normal[2] = normal[2];

  Vec<Float,3> t_point;
  t_point[0] = point[0];
  t_point[1] = point[1];
  t_point[2] = point[2];

  Vec<Float,3> *points_ptr = points.get_device_ptr();

  const Ray *ray_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray &ray = ray_ptr[i];
    const Float denom = dot(ray.m_dir, t_normal);
    Float dist = infinity<Float>();
    if(denom > 1e-6)
    {
      Vec<Float,3> p = t_point - ray.m_orig;
      const Float t = dot(p, t_normal) / denom;
      if(t > 0)
      {
        dist = t;
      }
    }

    Vec<Float,3> sample = ray.m_dir * dist + ray.m_orig;

    points_ptr[i] = sample;

  });

  return points;
}

} // namespace detail

SlicePlane::SlicePlane(DataSet &data_set)
  : Traceable(data_set)
{
  m_point[0] = 0.f;
  m_point[1] = 0.f;
  m_point[2] = 0.f;

  m_normal[0] = 0.f;
  m_normal[1] = 1.f;
  m_normal[2] = 0.f;
}

struct Functor
{
  SlicePlane *m_slice;
  Array<Ray> *m_rays;
  Array<RayHit> m_hits;
  Functor(SlicePlane *slice,
          Array<Ray> *rays)
    : m_slice(slice),
      m_rays(rays)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    m_slice->execute(topo.mesh(), *m_rays);
  }
};

Array<RayHit>
SlicePlane::nearest_hit(Array<Ray> &rays)
{
  TopologyBase *topo = m_data_set.topology();

  Functor func(this, &rays);
  dispatch_3d(topo, func);
  return func.m_hits;
}

template<class MeshElement>
Array<RayHit>
SlicePlane::execute(Mesh<MeshElement> &mesh, Array<Ray> &rays)
{
  DRAY_LOG_OPEN("slice_plane");

  Array<Vec<Float,3>> samples = detail::calc_sample_points(rays, m_point, m_normal);

  // Find elements and reference coordinates for the points.
  Array<Location> locations = mesh.locate(samples);

  Array<RayHit> hits = detail::get_hits(rays, locations, samples);

  DRAY_LOG_CLOSE();
  return hits;
}

void
SlicePlane::set_point(const Vec<float32,3> &point)
{
  m_point = point;
}

void
SlicePlane::set_normal(const Vec<float32,3> &normal)
{
  m_normal = normal;
  m_normal.normalize();
}

}}//namespace dray::ray_tracing


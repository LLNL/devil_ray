#ifndef DRAY_RAY_HPP
#define DRAY_RAY_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
// TODO: includes that need to be moved to ray ops
#include <dray/policies.hpp>
#include <dray/aabb.hpp>

namespace dray
{

template<typename T>
class Ray
{
public:
  Vec<T,3> m_dir;
  Vec<T,3> m_orig;
  T        m_near;
  T        m_far;
  T        m_dist;
  int32    m_pixel_id;
  int32    m_hit_idx;
  Vec<T,3> m_hit_ref_pt;

  int32    m_active;

#ifdef DRAY_STATS
  int32    m_wasted_steps;
  int32    m_total_steps;
#endif

  //static Ray gather_rays(const Ray rays, const Array<int32> indices);
};

template<typename T>
std::ostream & operator << (std::ostream &out, const Ray<T> &r)
{
  out<<r.m_pixel_id;
  return out;
}

template<typename T>
Array<Vec<T,3>> calc_tips(const Array<Ray<T>> &rays);

template<typename T>
Array<int32> active_indices(const Array<Ray<T>> &rays);

#ifdef DRAY_STATS
template <typename T>
void reset_step_counters(Array<Ray<T>> &rays);
#endif

typedef Ray<float32> ray32;
typedef Ray<float64> ray64;

template<typename T>
void advance_ray(Array<Ray<T>> &rays, float32 distance)
{
  // avoid lambda capture issues
  T dist = distance;

  Ray<T> *ray_ptr = rays.get_device_ptr();

  const int32 size = rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Ray<T> &ray = ray_ptr[i];
    // advance ray
    ray.m_dist += dist;
  });
}

//
// utility function to find estimated intersection with the bounding
// box of the mesh
//
// After calling:
//   rays m_near   : set to estimated mesh entry
//   rays m_far    : set to estimated mesh exit
//   rays hit_idx  : -1 if ray missed the AABB and 1 if it hit
//
//   if ray missed then m_far <= m_near, if ray hit then m_far > m_near.
//
template<typename T>
void calc_ray_start(Array<Ray<T>> &rays, AABB bounds)
{
  // avoid lambda capture issues
  AABB mesh_bounds = bounds;

  Ray<T> *ray_ptr = rays.get_device_ptr();

  const int32 size = rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Ray<T> ray = ray_ptr[i];
    const Vec<T,3> ray_dir = ray.m_dir;
    const Vec<T,3> ray_orig = ray.m_orig;

    float32 dirx = static_cast<float32>(ray_dir[0]);
    float32 diry = static_cast<float32>(ray_dir[1]);
    float32 dirz = static_cast<float32>(ray_dir[2]);
    float32 origx = static_cast<float32>(ray_orig[0]);
    float32 origy = static_cast<float32>(ray_orig[1]);
    float32 origz = static_cast<float32>(ray_orig[2]);

    const float32 inv_dirx = rcp_safe(dirx);
    const float32 inv_diry = rcp_safe(diry);
    const float32 inv_dirz = rcp_safe(dirz);

    const float32 odirx = origx * inv_dirx;
    const float32 odiry = origy * inv_diry;
    const float32 odirz = origz * inv_dirz;

    const float32 xmin = mesh_bounds.m_x.min() * inv_dirx - odirx;
    const float32 ymin = mesh_bounds.m_y.min() * inv_diry - odiry;
    const float32 zmin = mesh_bounds.m_z.min() * inv_dirz - odirz;
    const float32 xmax = mesh_bounds.m_x.max() * inv_dirx - odirx;
    const float32 ymax = mesh_bounds.m_y.max() * inv_diry - odiry;
    const float32 zmax = mesh_bounds.m_z.max() * inv_dirz - odirz;

    const float32 min_int = 0.f;
    float32 min_dist = max(max(max(min(ymin, ymax), min(xmin, xmax)), min(zmin, zmax)), min_int);
    float32 max_dist = min(min(max(ymin, ymax), max(xmin, xmax)), max(zmin, zmax));

    int32 hit = -1; // miss flag

    ray.m_active = 0;
    if (max_dist > min_dist)
    {
      hit = 1;
      ray.m_active = 1;
    }

    ray.m_near = min_dist;
    ray.m_dist = min_dist;
    ray.m_far = max_dist;

    ray.m_hit_idx = hit;
    ray_ptr[i] = ray;
  });
}

} // namespace dray
#endif

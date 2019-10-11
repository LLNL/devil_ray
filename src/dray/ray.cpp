#include <dray/ray.hpp>
#include <dray/array_utils.hpp>
#include <dray/policies.hpp>

namespace dray
{

Array<Vec<Float,3>> calc_tips(const Array<Ray> &rays)
{
  const int32 ray_size= rays.size();

  Array<Vec<Float,3>> tips;
  tips.resize(ray_size);

  //const Vec<T,3> *orig_ptr = m_orig.get_device_ptr_const();
  //const Vec<T,3> *dir_ptr = m_dir.get_device_ptr_const();
  //const T *dist_ptr = m_dist.get_device_ptr_const();
  const Ray * ray_ptr = rays.get_device_ptr_const();

  Vec<Float,3> *tips_ptr = tips.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 ii)
  {
    Ray ray = ray_ptr[ii];
    tips_ptr[ii] = ray.m_orig + ray.m_dir * ray.m_dist;
  });

  return tips;
}

Array<int32> active_indices(const Array<Ray> &rays)
{
  const int32 ray_size= rays.size();
  Array<int32> active_flags;
  active_flags.resize(ray_size);
  const Ray * ray_ptr = rays.get_device_ptr_const();

  int32 *flags_ptr = active_flags.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 ii)
  {
    uint8 flag = ((ray_ptr[ii].m_active > 0) &&
                  (ray_ptr[ii].m_near < ray_ptr[ii].m_far) &&
                  (ray_ptr[ii].m_dist < ray_ptr[ii].m_far)) ? 1 : 0;
    flags_ptr[ii] = flag;
  });

  // TODO: we can do this without this: have index just look at the index
  Array<int32> idxs = array_counting(ray_size, 0,1);

  return index_flags(active_flags, idxs);
}

void advance_ray(Array<Ray> &rays, float32 distance)
{
  // avoid lambda capture issues
  Float dist = distance;

  Ray *ray_ptr = rays.get_device_ptr();

  const int32 size = rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Ray &ray = ray_ptr[i];
    // advance ray
    ray.m_dist += dist;
  });
}

//template<typename T>
//Array<Ray<T>> gather_rays(const Ray<T> i_rays, const Array<int32> indices)
//{
//  Ray<T> o_rays;
//
//  o_rays.m_dir = gather(i_rays.m_dir, indices);
//  o_rays.m_orig = gather(i_rays.m_orig, indices);
//  o_rays.m_near = gather(i_rays.m_near, indices);
//  o_rays.m_far = gather(i_rays.m_far, indices);
//  o_rays.m_dist = gather(i_rays.m_dist, indices);
//  o_rays.m_pixel_id = gather(i_rays.m_pixel_id, indices);
//  o_rays.m_hit_idx = gather(i_rays.m_hit_idx, indices);
//  o_rays.m_hit_ref_pt = gather(i_rays.m_hit_ref_pt, indices);
//
//  return o_rays;
//}

void calc_ray_start(Array<Ray> &rays, AABB<> bounds)
{
  // avoid lambda capture issues
  AABB<> mesh_bounds = bounds;
  // be conservative
  mesh_bounds.scale(1.001f);

  Ray *ray_ptr = rays.get_device_ptr();

  const int32 size = rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Ray ray = ray_ptr[i];
    const Vec<Float,3> ray_dir = ray.m_dir;
    const Vec<Float,3> ray_orig = ray.m_orig;

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

    const float32 xmin = mesh_bounds.m_ranges[0].min() * inv_dirx - odirx;
    const float32 ymin = mesh_bounds.m_ranges[1].min() * inv_diry - odiry;
    const float32 zmin = mesh_bounds.m_ranges[2].min() * inv_dirz - odirz;
    const float32 xmax = mesh_bounds.m_ranges[0].max() * inv_dirx - odirx;
    const float32 ymax = mesh_bounds.m_ranges[1].max() * inv_diry - odiry;
    const float32 zmax = mesh_bounds.m_ranges[2].max() * inv_dirz - odirz;

    const float32 min_int = 0.f;
    float32 min_dist = max(max(max(min(ymin, ymax), min(xmin, xmax)), min(zmin, zmax)), min_int);
    float32 max_dist = min(min(max(ymin, ymax), max(xmin, xmax)), max(zmin, zmax));

    ray.m_active = 0;
    if (max_dist > min_dist)
    {
      ray.m_active = 1;
    }

    ray.m_near = min_dist;
    ray.m_dist = min_dist;
    ray.m_far = max_dist;

    ray_ptr[i] = ray;
  });
}
} // namespace dray

#include <dray/ray.hpp>
#include <dray/array_utils.hpp>
#include <dray/policies.hpp>

namespace dray
{

//template <typename T>
//void Ray<T>::reactivate()
//{
//  m_active_rays = array_counting(size(), 0,1);
//}


template<typename T>
Array<Vec<T,3>> calc_tips(const Array<Ray<T>> &rays)
{
  const int32 ray_size= rays.size();

  Array<Vec<T,3>> tips;
  tips.resize(ray_size);

  //const Vec<T,3> *orig_ptr = m_orig.get_device_ptr_const();
  //const Vec<T,3> *dir_ptr = m_dir.get_device_ptr_const();
  //const T *dist_ptr = m_dist.get_device_ptr_const();
  const Ray<T> * ray_ptr = rays.get_device_ptr_const();

  Vec<T,3> *tips_ptr = tips.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 ii)
  {
    Ray<T> ray = ray_ptr[ii];
    tips_ptr[ii] = ray.m_orig + ray.m_dir * ray.m_dist;
  });

  return tips;
}

template<typename T>
Array<int32> active_indices(const Array<Ray<T>> &rays)
{
  const int32 ray_size= rays.size();
  Array<uint8> active_flags;
  active_flags.resize(ray_size);
  const Ray<T> * ray_ptr = rays.get_device_ptr_const();

  uint8 *flags_ptr = active_flags.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 ii)
  {
    uint8 flag = ray_ptr[ii].m_active > 0 ? 1 : 0;
    flags_ptr[ii] = flag;
  });

  // TODO: we can do this without this: have index just look at the index
  Array<int32> idxs = array_counting(ray_size, 0,1);

  return index_flags(active_flags, idxs);
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

template class Ray<float32>;
template class Ray<float64>;
template Array<Vec<float32,3>> calc_tips<float32>(const Array<Ray<float32>> &rays);
template Array<Vec<float64,3>> calc_tips<float64>(const Array<Ray<float64>> &rays);

template Array<int32> active_indices<float32>(const Array<Ray<float32>> &rays);
template Array<int32> active_indices<float64>(const Array<Ray<float64>> &rays);

} // namespace dray

#include <dray/ray.hpp>
#include <dray/policies.hpp>

namespace dray
{

template<typename T>
void Ray<T>::resize(const int32 size)
{
  m_dir.resize(size);
  m_orig.resize(size);
  m_near.resize(size);
  m_far.resize(size);
  m_dist.resize(size);
  m_pixel_id.resize(size);
  m_hit_idx.resize(size);
  m_hit_ref_pt.resize(size);
}

template<typename T>
int32 Ray<T>::size() const
{
  return m_dir.size();
}


template<typename T>
Array<Vec<T,3>> Ray<T>::calc_tips() const
{
  const int32 ray_size = size();

  Array<Vec<T,3>> tips;
  tips.resize(ray_size);

  const Vec<T,3> *orig_ptr = m_orig.get_device_ptr_const();
  const Vec<T,3> *dir_ptr = m_dir.get_device_ptr_const();
  const T *dist_ptr = m_dist.get_device_ptr_const();

  Vec<T,3> *tips_ptr = tips.get_device_ptr();
  
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 ii)
  {
    tips_ptr[ii] = orig_ptr[ii] + dir_ptr[ii] * dist_ptr[ii];
  });

  return tips;
}

template class Ray<float32>;
template class Ray<float64>;

} // namespace dray

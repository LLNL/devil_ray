#include <dray/ray.hpp>

namespace dray
{

template<typename T>
void Ray<T>::resize(const int32 size)
{
  m_dir.resize(size);
  m_orig.resize(size);
  m_near.resize(size);
  m_far.resize(size);
  m_pixel_id.resize(size);
  m_hit_idx.resize(size);
}

template<typename T>
int32 Ray<T>::size()
{
  return m_dir.size();
}

template class Ray<float32>;
template class Ray<float64>;

} // namespace dray

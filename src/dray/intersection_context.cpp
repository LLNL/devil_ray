#include <dray/intersection_context.hpp>

namespace dray
{

template<typename T>
void IntersectionContext<T>::resize(const int32 size)
{
  m_is_valid.resize(size);
  m_hit_pt.resize(size);
  m_normal.resize(size);
  m_ray_dir.resize(size);
  m_pixel_id.resize(size);
}

template<typename T>
int32 IntersectionContext<T>::size() const
{
  return m_hit_pt.size();
}

template class IntersectionContext<float32>;
template class IntersectionContext<float64>;

} // namespace dray

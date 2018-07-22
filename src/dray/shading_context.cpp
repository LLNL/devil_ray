#include <dray/shading_context.hpp>

namespace dray
{

template <typename T>
void
ShadingContext<T>::resize(const int32 size)
{
  m_is_valid.resize(size);
  m_hit_pt.resize(size);
  m_normal.resize(size);
  m_sample_val.resize(size);
  m_gradient_mag.resize(size);
  m_ray_dir.resize(size);
  m_pixel_id.resize(size);
}

template <typename T>
int32
ShadingContext<T>::size() const
{
  return m_sample_val.size();
}


template class ShadingContext<float32>;
template class ShadingContext<float64>;

} // namespace dray

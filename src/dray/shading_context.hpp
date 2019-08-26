#ifndef DRAY_SHADING_CONTEXT_HPP
#define DRAY_SHADING_CONTEXT_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/types.hpp>

namespace dray
{

template<typename T>
class ShadingContext
{
public:
  int32    m_is_valid;
  Vec<T,3> m_hit_pt;
  Vec<T,3> m_normal;
  T        m_sample_val;
  T        m_gradient_mag;
  Vec<T,3> m_ray_dir;
  int32    m_pixel_id;

};

template<typename T>
std::ostream & operator << (std::ostream &out, const ShadingContext<T> &r)
{
  out<<r.m_pixel_id;
  return out;
}

} // namespace dray

#endif

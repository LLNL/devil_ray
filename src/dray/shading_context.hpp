#ifndef DRAY_SHADING_CONTEXT_HPP
#define DRAY_SHADING_CONTEXT_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/types.hpp>

namespace dray
{

class ShadingContext
{
public:
  int32        m_is_valid;
  Vec<Float,3> m_hit_pt;
  Vec<Float,3> m_normal;
  Float        m_sample_val;
  Float        m_gradient_mag;
  Vec<Float,3> m_ray_dir;
  int32        m_pixel_id;

};

std::ostream & operator << (std::ostream &out, const ShadingContext &r);

} // namespace dray

#endif

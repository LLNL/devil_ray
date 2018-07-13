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
  Array<int32>    m_is_valid;
  Array<Vec<T,3>> m_hit_pt;
  Array<Vec<T,3>> m_normal;
  Array<T>  m_sample_val;
  Array<Vec<T,3>> m_ray_dir;
  Array<int32>    m_pixel_id;

  void resize(const int32 size);

  int32 size() const;
};

} // namespace dray

#endif

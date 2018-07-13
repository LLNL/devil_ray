#ifndef DRAY_INTERSECTION_CONTEXT_HPP
#define DRAY_INTERSECTION_CONTEXT_HPP

#include <dray/array.hpp>
//#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

template<typename T>
class IntersectionContext
{
public:
  Array<int32>    m_is_valid;
  Array<Vec<T,3>> m_hit_pt;
  Array<Vec<T,3>> m_normal;
  Array<Vec<T,3>> m_ray_dir;
  Array<int32>    m_pixel_id;

  void resize(const int32 size);
  int32 size() const;
};

} // namespace dray
#endif

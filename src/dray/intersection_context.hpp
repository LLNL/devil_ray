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
  int32    m_is_valid;
  Vec<T,3> m_hit_pt;
  Vec<T,3> m_normal;
  Vec<T,3> m_ray_dir;
  int32    m_pixel_id;
};

template<typename T>
std::ostream & operator << (std::ostream &out, const IntersectionContext<T> &r)
{
  out<<r.m_pixel_id;
  return out;
}

} // namespace dray
#endif

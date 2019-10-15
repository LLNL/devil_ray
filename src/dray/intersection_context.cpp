#include <dray/intersection_context.hpp>

namespace dray
{

std::ostream & operator << (std::ostream &out, const IntersectionContext &r)
{
  out<<r.m_pixel_id;
  return out;
}

} // namespace dray

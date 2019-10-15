#include <dray/shading_context.hpp>

namespace dray
{

std::ostream & operator << (std::ostream &out, const ShadingContext &r)
{
  out<<r.m_pixel_id;
  return out;
}

} // namespace dray

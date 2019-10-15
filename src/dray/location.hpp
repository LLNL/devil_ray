#ifndef DRAY_LOCATION_HPP
#define DRAY_LOCATION_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class Location
{
public:
  Vec<Float,3> m_ref_pt;
  int32 m_cell_id;
};

static
std::ostream & operator << (std::ostream &out, const RayHit &hit)
{
  out<<"[ cell_id "<<hit.m_cell_id<<" ref_pt: "<<m_ref_pt<<" ]";
  return out;
}

} // namespace dray
#endif

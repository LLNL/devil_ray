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
  int32        m_cell_id; /*!< Cell containing the location. -1 indicates not found */
  Vec<Float,3> m_ref_pt;  /*!< Refence space coordinates of location */
};

static
std::ostream & operator << (std::ostream &out, const RayHit &hit)
{
  out<<"[ cell_id "<<hit.m_cell_id<<" ref_pt: "<<m_ref_pt<<" ]";
  return out;
}

} // namespace dray
#endif

#ifndef DRAY_RAY_HIT_HPP
#define DRAY_RAY_HIT_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class RayHit
{
public:
  int32 m_hit_idx;
  Float m_dist;
  int32 m_cell_id;
};

static
std::ostream & operator << (std::ostream &out, const RayHit &hit)
{
  out<<"[ hit_idx: "<<hit.m_hit_idx<<" dist: "<<hit.m_dist<<" cell_id "<<hit.m_cell_id<<" ]";
  return out;
}

} // namespace dray
#endif

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
  int32        m_hit_idx;  /*!< Hit index of primitive hit by ray. -1 means miss */
  Float        m_dist;     /*!< Distance to the hit */
  Vec<Float,3> m_ref_pt;   /*!< Refence space coordinates of hit */
};

static
std::ostream & operator << (std::ostream &out, const RayHit &hit)
{
  out<<"[ hit_idx: "<<hit.m_hit_idx<<" dist: "<<hit.m_dist<<" ref "<<hit.m_ref_pt<<" ]";
  return out;
}

} // namespace dray
#endif

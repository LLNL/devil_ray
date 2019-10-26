#ifndef DRAY_FRAGMENT_HPP
#define DRAY_FRAGMENT_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class Fragment
{
public:
  float32 m_scalar;  /*!< Hit index of primitive hit by ray. -1 means miss */
  Vec<float32,3> m_normal;   /*!< Refence space coordinates of hit */
};

static
std::ostream & operator << (std::ostream &out, const Fragment &frag)
{
  out<<"[ scalar : "<<frag.m_scalar<<" norm: "<<frag.m_normal<<" ]";
  return out;
}

} // namespace dray
#endif

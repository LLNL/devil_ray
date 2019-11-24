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
  float32 m_scalar; /*!< non-normalized scalar value */
  Vec<float32, 3> m_normal; /*!< non-normalized surface normal or scalar gradient */
};

static std::ostream &operator<< (std::ostream &out, const Fragment &frag)
{
  out << "[ scalar : " << frag.m_scalar << " norm: " << frag.m_normal << " ]";
  return out;
}

} // namespace dray
#endif

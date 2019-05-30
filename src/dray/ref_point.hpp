#ifndef DRAY_REF_POINT_HPP
#define DRAY_REF_POINT_HPP

#include <dray/vec.hpp>

namespace dray
{
  template <typename T, int32 dim = 3>
  struct RefPoint
  {
    int32 m_el_id;
    Vec<T,dim> m_el_coords;
  };

  template<typename T, int32 dim>
  std::ostream & operator << (std::ostream &out, const RefPoint<T,dim> &rpt)
  {
    out<<rpt.m_el_id;
    return out;
  }

}

#endif//DRAY_REF_POINT_HPP

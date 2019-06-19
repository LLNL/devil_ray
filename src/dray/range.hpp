#ifndef DRAY_RANGE_HPP
#define DRAY_RANGE_HPP

#include <dray/exports.hpp>
#include <dray/math.hpp>

#include <iostream>

namespace dray
{

class Range 
{
protected:
  float32 m_min = infinity32();
  float32 m_max = neg_infinity32();
public:

  // Not using these to remain POD
  //DRAY_EXEC Range()
  //  : m_min(infinity32()),
  //    m_max(neg_infinity32())
  //{
  //}

  //template<typename T1, typename T2>
  //DRAY_EXEC Range(const T1 &min, const T2 &max)
  //  : m_min(float32(min)),
  //    m_max(float32(max))
  //{
  //}

  DRAY_EXEC void reset()
  {
    m_min = infinity32();
    m_max = neg_infinity32();
  }

  DRAY_EXEC
  float32 min() const
  {
    return m_min;
  }

  DRAY_EXEC
  float32 max() const
  {
    return m_max;
  }

  DRAY_EXEC
  bool is_empty() const
  {
    return m_min > m_max;
  }
    
  template<typename T>
  DRAY_EXEC
  void include(const T &val)
  {
    m_min = fmin(m_min, float32(val));
    m_max = fmax(m_max, float32(val));
  }

  DRAY_EXEC
  void include(const Range &other)
  {
    if(!other.is_empty())
    {
      include(other.min());
      include(other.max());
    }
  }

  DRAY_EXEC
  void intersect(const Range &other)
  {
    m_min = fmax(m_min, other.min());
    m_max = fmin(m_max, other.max());
  }

  DRAY_EXEC
  Range identity() const
  {
    return Range();
  }

  DRAY_EXEC
  static Range mult_identity()
  {
    Range ret;
    ret.m_min = neg_infinity32();
    ret.m_max = infinity32();
    return ret;
  }

  DRAY_EXEC
  float32 center() const
  {
    if(is_empty())
    {
      return nan32();
    }
    else return 0.5f * (m_min + m_max);
  }

  DRAY_EXEC
  void split(float32 alpha, Range &left, Range &right) const
  {
    left.m_min = m_min;
    right.m_max = m_max;

    left.m_max = right.m_min = m_min * (1.0 - alpha) + m_max * alpha;
  }

  DRAY_EXEC
  float32 length() const
  {
    if(is_empty())
    {
      return nan32();
    }
    else return m_max - m_min;
  }

  DRAY_EXEC
  void scale(float32 scale)
  {
    if(is_empty())
    {
      return;
    }

    float32 c = center();
    float32 delta = scale * 0.5f * length();
    include(c - delta);
    include(c + delta);
  }


  DRAY_EXEC
  Range operator+(const Range &other) const
  {
    Range res;
    res.include(*this);
    res.include(other);
    return res;
  }


  friend std::ostream& operator<<(std::ostream &os, const Range &range);
  
};

inline std::ostream& operator<<(std::ostream &os, const Range &range)
{
  os<<"[";
  os<<range.min()<<", ";
  os<<range.max();
  os<<"]";
  return os;
}

} // namespace dray
#endif

#ifndef DRAY_RANGE_HPP
#define DRAY_RANGE_HPP

#include <dray/exports.hpp>
#include <dray/math.hpp>

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
  Range identity() const
  {
    return Range();
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
  Range operator+(const Range &other) const
  {
    Range res;
    res.include(*this);
    res.include(other);
    return res;
  }


  
};

} // namespace dray
#endif

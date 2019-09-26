#ifndef DRAY_RANGE_HPP
#define DRAY_RANGE_HPP

#include <dray/exports.hpp>
#include <dray/math.hpp>

#include <iostream>
#include <assert.h>

namespace dray
{

template <typename F = float32>
class Range;

template <typename F = float32>
inline std::ostream& operator<<(std::ostream &os, const Range<F> &range);

template <typename F>
class Range
{
protected:
  F m_min = infinity32();
  F m_max = neg_infinity32();
public:

  // Not using these to remain POD
  //DRAY_EXEC Range()
  //  : m_min(infinity32()),
  //    m_max(neg_infinity32())
  //{
  //}

  //template<typename T1, typename T2>
  //DRAY_EXEC Range(const T1 &min, const T2 &max)
  //  : m_min(F(min)),
  //    m_max(F(max))
  //{
  //}

  DRAY_EXEC void reset()
  {
    m_min = infinity32();
    m_max = neg_infinity32();
  }

  DRAY_EXEC
  F min() const
  {
    return m_min;
  }

  DRAY_EXEC
  F max() const
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
    m_min = fmin(m_min, F(val));
    m_max = fmax(m_max, F(val));
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
  bool is_contained_in(const Range &other)
  {
    return is_empty() || (other.m_min <= m_min && m_max <= other.m_max);
  }

  DRAY_EXEC
  bool contains(const Range &other)
  {
    return other.is_empty() || (m_min <= other.m_min && other.m_max <= m_max);
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
  static Range ref_universe()
  {
    Range ret;
    ret.m_min = 0.f;
    ret.m_max = 1.0;
    return ret;
  }

  DRAY_EXEC
  F center() const
  {
    if(is_empty())
    {
      return nan32();
    }
    else return 0.5f * (m_min + m_max);
  }

  DRAY_EXEC
  void split(F alpha, Range &left, Range &right) const
  {
    left.m_min = m_min;
    right.m_max = m_max;

    left.m_max = right.m_min = m_min * (1.0 - alpha) + m_max * alpha;
  }

  DRAY_EXEC
  F length() const
  {
    if(is_empty())
    {
      // should this just return 0?
      return nan32();
    }
    else return m_max - m_min;
  }

  DRAY_EXEC
  void scale(F scale)
  {
    if(is_empty())
    {
      return;
    }

    F c = center();
    F delta = scale * 0.5f * length();
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

  DRAY_EXEC
  Range intersect(const Range &other) const
  {
    Range res;
    res.m_min = ::max(m_min, other.m_min);
    res.m_max = ::min(m_max, other.m_max);

    return res;
  }

  DRAY_EXEC
  Range split()
  {
    assert(!is_empty());
    Range other_half(*this);
    const float32 mid = center();
    m_min = mid;
    other_half.m_max = mid;
    return other_half;
  }


  friend std::ostream& operator<< <F>(std::ostream &os, const Range &range);

};

template <typename F>
inline std::ostream& operator<<(std::ostream &os, const Range<F> &range)
{
  os<<"[";
  os<<range.min()<<", ";
  os<<range.max();
  os<<"]";
  return os;
}

} // namespace dray
#endif

#ifndef DRAY_AABB_HPP
#define DRAY_AABB_HPP

//#include <dray/exports.hpp>
#include <dray/range.hpp>
#include <dray/vec.hpp>

#include <iostream>

namespace dray
{

template <int32 dim> class AABB;

template <int32 dim>
inline std::ostream& operator<<(std::ostream &os, const AABB<dim> &range);

template <int32 dim = 3>
class AABB
{

public:
  Range<> m_ranges[dim];

  DRAY_EXEC
  void include(const AABB &other)
  {
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].include(other.m_ranges[d]);
  }
  
  DRAY_EXEC
  void include(const Vec<float32, dim> &point)
  {
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].include(point[d]);
  }

  DRAY_EXEC
  void intersect(const AABB &other)
  {
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].intersect(other.m_ranges[d]);
  }
 
  DRAY_EXEC
  void expand(const float32 &epsilon)
  {
    assert(epsilon > 0.f);
    for (int32 d = 0; d < dim; d++)
    {
      m_ranges[d].include(m_ranges[d].min() - epsilon);
      m_ranges[d].include(m_ranges[d].max() + epsilon);
    }
  }

  DRAY_EXEC
  void scale(const float32 &scale)
  {
    assert(scale >= 1.f);
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].scale(scale);
  }

  DRAY_EXEC
  Vec<float32, dim> center() const
  {
    Vec<float32, dim> center;
    for (int32 d = 0; d < dim; d++)
      center[d] = m_ranges[d].center();
    return center;
  }

  // Mins of all of the ranges.
  DRAY_EXEC
  Vec<float32, dim> min() const
  {
    Vec<float32, dim> lower_left;
    for (int32 d = 0; d < dim; d++)
      lower_left[d] = m_ranges[d].min();
    return lower_left;
  }

  // Maxes of all the ranges.
  DRAY_EXEC
  Vec<float32, dim> max() const
  {
    Vec<float32, dim> upper_right;
    for (int32 d = 0; d < dim; d++)
      upper_right[d] = m_ranges[d].max();
    return upper_right;
  }

  DRAY_EXEC
  static AABB universe()
  {
    AABB universe;
    for (int32 d = 0; d < dim; d++)
      universe.m_ranges[d] = Range<>::mult_identity();
    return universe;
  }
 
  //DRAY_EXEC
  //AABB identity() const
  //{
  //  return AABB();
  //}

  //DRAY_EXEC
  //AABB operator+(const AABB &other) const
  //{
  //  AABB res = *this;
  //  res.include(other);
  //  return res;
  //}

  friend std::ostream& operator<< <dim> (std::ostream &os, const AABB &aabb);
};


template <int32 dim>
inline std::ostream& operator<<(std::ostream &os, const AABB<dim> &aabb)
{
  os << aabb.min() << " - " << aabb.max();
  return os;
}

} // namespace dray
#endif

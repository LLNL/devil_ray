#ifndef DRAY_AABB_HPP
#define DRAY_AABB_HPP

//#include <dray/exports.hpp>
#include <dray/range.hpp>
#include <dray/vec.hpp>

#include <iostream>

namespace dray
{

class AABB
{

public:
  Range m_x; 
  Range m_y; 
  Range m_z; 

  DRAY_EXEC
  void include(const AABB &other)
  {
    m_x.include(other.m_x); 
    m_y.include(other.m_y); 
    m_z.include(other.m_z); 
  }
  
  DRAY_EXEC
  void include(const Vec3f &point)
  {
    m_x.include(point[0]); 
    m_y.include(point[1]); 
    m_z.include(point[2]); 
  }
 
  DRAY_EXEC
  void expand(const float32 &epsilon)
  {
    assert(epsilon > 0.f);
    m_x.include(m_x.min() - epsilon); 
    m_x.include(m_x.max() + epsilon); 
    m_y.include(m_y.min() - epsilon); 
    m_y.include(m_y.max() + epsilon); 
    m_z.include(m_z.min() - epsilon); 
    m_z.include(m_z.max() + epsilon); 
  }

  DRAY_EXEC
  void scale(const float32 &scale)
  {
    assert(scale >= 1.f);
    m_x.scale(scale);
    m_y.scale(scale);
    m_z.scale(scale);
  }

  DRAY_EXEC
  Vec3f center() const
  {
    return make_vec3f(m_x.center(),
                      m_y.center(),
                      m_z.center());
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

  friend std::ostream& operator<<(std::ostream &os, const AABB &aabb);
};

inline std::ostream& operator<<(std::ostream &os, const AABB &aabb)
{
  os<<"[";
  os<<aabb.m_x.min()<<", ";
  os<<aabb.m_y.min()<<", ";
  os<<aabb.m_z.min();
  os<<"] - [";
  os<<aabb.m_x.max()<<", ";
  os<<aabb.m_y.max()<<", ";
  os<<aabb.m_z.max();
  os<<"]";
  return os;
}

} // namespace dray
#endif

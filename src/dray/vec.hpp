#ifndef DRAY_VEC_HPP
#define DRAY_VEC_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>

#include <assert.h>
#include <iostream>

#ifdef __CUDACC__
#include "math.h"
#else
#include <math.h>
#endif

namespace dray 
{

template<typename T, int32 S>
class Vec
{

public:
  T m_data[S];

  //
  //  No contructors so this is a POD type
  //

  DRAY_EXEC bool operator==(const Vec<T,S> &other) const
  {
    bool e = true;
    for(int i = 0; i < S; ++i)
    {
      if(m_data[i] != other[i]) e = false;
    }
    return e;
  }

  template<typename TT, int32 SS>
  friend std::ostream& operator<<(std::ostream &os, const Vec<TT,SS> &vec);

  DRAY_EXEC void operator=(const Vec<T,S> &other)
  {
    for(int i = 0; i < S; ++i)
    {
      m_data[i] = other.m_data[i];
    }
  }

  DRAY_EXEC void operator=(const T &single_val)
  {
    for(int i = 0; i < S; ++i)
    {
      m_data[i] = single_val;
    }
  }

  //DRAY_EXEC const T operator[](const int32 &i) const
  //{
  //  assert(i > -1 && i < S);
  //  return m_data[i];
  //}

  DRAY_EXEC const T& operator[](const int32 &i) const
  {
    assert(i > -1 && i < S);
    return m_data[i];
  }

  DRAY_EXEC T& operator[](const int32 &i)
  {
    assert(i > -1 && i < S);
    return m_data[i];
  }

  // scalar mult /  div
  DRAY_EXEC Vec<T,S> operator*(const T &s) const
  {
    Vec<T,S> res;

    for(int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] * s; 
    }

    return res;
  }

  DRAY_EXEC Vec<T,S> operator/(const T &s) const
  {
    Vec<T,S> res;

    for(int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] / s; 
    }

    return res;
  }

  DRAY_EXEC void operator*=(const T &s)
  {
    for(int i = 0; i < S; ++i)
    {
      m_data[i] *= s; 
    }
  }

  DRAY_EXEC void operator/=(const T &s)
  {
    for(int i = 0; i < S; ++i)
    {
      m_data[i] /= s; 
    }

  }

  // vector add / sub

  DRAY_EXEC Vec<T,S> operator+(const Vec<T,S> &other) const
  {
    Vec<T,S> res;

    for(int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] + other[i]; 
    }

    return res;
  }

  DRAY_EXEC Vec<T,S> operator-(const Vec<T,S> &other) const
  {
    Vec<T,S> res;

    for(int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] - other[i]; 
    }

    return res;
  }

  DRAY_EXEC void operator+=(const Vec<T,S> &other)
  {

    for(int i = 0; i < S; ++i)
    {
      m_data[i] += other[i]; 
    }

  }

  DRAY_EXEC void operator-=(const Vec<T,S> &other)
  {

    for(int i = 0; i < S; ++i)
    {
      m_data[i] -= other[i]; 
    }

  }

  DRAY_EXEC Vec<T,S> operator-(void) const
  {
    Vec<T,S> res;

    for(int i = 0; i < S; ++i)
    {
      res[i] = -m_data[i]; 
    }

    return res;
  }

  DRAY_EXEC T magnitude() const
  {
    T sum = T(0);

    for(int i = 0; i < S; ++i)
    {
      sum += m_data[i] * m_data[i]; 
    }

    return sqrtf(sum); 
  }


  DRAY_EXEC void normalize()
  {
    T mag = magnitude();
    *this /= mag;
  }

  DRAY_EXEC T Normlinf() const   // Used for convergence tests.
  {
    // Max{ abs(x_i) } over all components.
    T max_c = max(-m_data[0], m_data[0]);
    for (int ii = 1; ii < S; ++ii)
    {
      max_c = max( max_c, max(-m_data[ii], m_data[ii]) );
    }
    return max_c;
  }

};

// vector utility functions
template<typename T, int32 S>
DRAY_EXEC T dot(const Vec<T,S> &a, const Vec<T,S> &b)
{
  T res = T(0);

  for(int i = 0; i < S; ++i)
  {
    res += a[i] * b[i]; 
  }

  return res;
}

template<typename T>
DRAY_EXEC Vec<T,3> cross(const Vec<T,3> &a, const Vec<T,3> &b)
{
  Vec<T,3> res; 
  res[0] = a[1] * b[2] - a[2] * b[1];
  res[1] = a[2] * b[0] - a[0] * b[2];
  res[2] = a[0] * b[1] - a[1] * b[0];
  return res;
}

template<typename TT, int32 SS>
std::ostream& operator<<(std::ostream &os, const Vec<TT,SS> &vec)
{
  os<<"[";
  for(int i = 0; i < SS; ++i)
  {
    os<<vec[i]; 
    if(i != SS - 1) os<<", ";
  }
  os<<"]";
  return os;
}

// typedefs
typedef Vec<int32,2> Vec2i;
typedef Vec<int64,2> Vec2li;
typedef Vec<float32,2> Vec2f;
typedef Vec<float64,2> Vec2d;

typedef Vec<int32,3> Vec3i;
typedef Vec<int64,3> Vec3li;
typedef Vec<float32,3> Vec3f;
typedef Vec<float64,3> Vec3d;

typedef Vec<int32,4> Vec4i;
typedef Vec<int64,4> Vec4li;
typedef Vec<float32,4> Vec4f;
typedef Vec<float64,4> Vec4d;

DRAY_EXEC
Vec2i make_vec2i(const int32 &a, const int32 &b)
{
  Vec2i res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec2li make_vec2li(const int64 &a, const int64 &b)
{
  Vec2li res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec2f make_vec2f(const float32 &a, const float32 &b)
{
  Vec2f res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec2d make_vec2d(const float64 &a, const float64 &b)
{
  Vec2d res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec3i make_vec3i(const int32 &a, const int32 &b, const int32 &c)
{
  Vec3i res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec3li make_vec3li(const int64 &a, const int64 &b, const int64 &c)
{
  Vec3li res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec3f make_vec3f(const float32 &a, const float32 &b, const float32 &c)
{
  Vec3f res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec3d make_vec3d(const float64 &a, const float64 &b, const float64 &c)
{
  Vec3d res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec4i make_vec4i(const int32 &a, const int32 &b, const int32 &c, const int32 &d)
{
  Vec4i res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}

DRAY_EXEC
Vec4li make_vec4li(const int64 &a, const int64 &b, const int64 &c, const int64 &d)
{
  Vec4li res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}

DRAY_EXEC
Vec4f make_vec4f(const float32 &a, const float32 &b, const float32 &c, const float32 &d)
{
  Vec4f res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}

DRAY_EXEC
Vec4d make_vec4d(const float64 &a, const float64 &b, const float64 &c, const float64 &d)
{
  Vec4d res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}

} // namespace dray

#endif

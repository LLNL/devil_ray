#ifndef DRAY_MATH_HPP
#define DRAY_MATH_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>

// include math so we can use functions defined
// in both cuda and c
#include <math.h>

#define DRAY_INF_32 0x7f800000U
#define DRAY_NG_INF_32 0xff800000U

#define DRAY_INF_64 0x7ff0000000000000ULL
#define DRAY_NG_INF_64 0xfff0000000000000ULL

#define DRAY_NAN_32 0x7FC00000U
#define DRAY_NAN_64 0x7FF8000000000000ULL

#ifndef __CUDACC__
// make sure min / max resolve for both cuda and cpu
#include <math.h>
#include <string.h> //resolve memcpy
using namespace std;
#endif

namespace dray
{
namespace detail
{

union Bits32 
{
  float32 scalar; 
  uint32  bits;
};

union Bits64 
{
  float64 scalar; 
  uint64  bits;
};

} // namespace detail

DRAY_EXEC 
float32 nan32()
{
  detail::Bits32 nan;
  nan.bits = DRAY_NAN_32;
  return nan.scalar;
}

DRAY_EXEC 
float32 infinity32()
{
  detail::Bits32 inf;
  inf.bits = DRAY_INF_32;
  return inf.scalar;
}

DRAY_EXEC 
float32 neg_infinity32()
{
  detail::Bits32 ninf;
  ninf.bits = DRAY_NG_INF_32;
  return ninf.scalar;
}

DRAY_EXEC 
float64 nan64()
{
  detail::Bits64 nan;
  nan.bits = DRAY_NAN_64;
  return nan.scalar;
}

DRAY_EXEC 
float64 infinity64()
{
  detail::Bits64 inf;
  inf.bits = DRAY_INF_64;
  return inf.scalar;
}

DRAY_EXEC 
float64 neg_infinity64()
{
  detail::Bits64 ninf;
  ninf.bits = DRAY_NG_INF_64;
  return ninf.scalar;
}

template<typename T> 
DRAY_EXEC
T infinity();

template<>
DRAY_EXEC
float32 infinity<float32>()
{
  return infinity32();
}

template<>
DRAY_EXEC
float64 infinity<float64>()
{
  return infinity64();
}

//
// count leading zeros
//
DRAY_EXEC
int32 clz(uint32 x)
{
  uint32 y;
  uint32 n = 32;
  y = x >> 16;
  if (y != 0)
  {
    n = n - 16;
    x = y;
  }
  y = x >> 8;
  if (y != 0)
  {
    n = n - 8;
    x = y;
  }
  y = x >> 4;
  if (y != 0)
  {
    n = n - 4;
    x = y;
  }
  y = x >> 2;
  if (y != 0)
  {
    n = n - 2;
    x = y;
  }
  y = x >> 1;
  if (y != 0)
    return int32(n - 2);
  return int32(n - x);
}

DRAY_EXEC
float64 pi()
{
  return 3.14159265358979323846264338327950288;
}

DRAY_EXEC
float32 rcp(float32 f) 
{ 
  return 1.0f / f; 
}

DRAY_EXEC
float64 rcp(float64 f)
{ 
  return 1.0 / f; 
}

DRAY_EXEC
float64 rcp_safe(float64 f)
{
  return rcp((fabs(f) < 1e-8) ? 1e-8 : f);
}

DRAY_EXEC
float32 rcp_safe(float32 f)
{
  return rcp((fabs(f) < 1e-8f) ? 1e-8f : f);
}

} // namespace dray
#endif

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

} // namespace dray
#endif

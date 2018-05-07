#ifndef DRAY_MATH_HPP
#define DRAY_MATH_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>

// include math so we can use functions defined
// in both cuda and c
#include <math.h>
#include <limits>
namespace dray
{

DRAY_EXEC 
float32 infinity32()
{
#if defined __CUDACC__
  return __int_as_float(0x7f800000);
#else
  return std::numeric_limits<float32>::infinity();
#endif
}

DRAY_EXEC 
float32 neg_infinity32()
{
#if defined __CUDACC__
  return __int_as_float(0xff800000);
#else
  return -std::numeric_limits<float32>::infinity();
#endif
}

DRAY_EXEC 
float64 infinity64()
{
#if defined __CUDACC__
  return __int_as_double(0x7ff0000000000000);
#else
  return std::numeric_limits<float32>::infinity();
#endif
}

DRAY_EXEC 
float64 neg_infinity64()
{
#if defined __CUDACC__
  return _int_as_double(0xfff0000000000000);
#else
  return -std::numeric_limits<float32>::infinity();
#endif
}

} // namespace dray
#endif

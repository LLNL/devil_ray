// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

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

#define DRAY_EPSILON_32 1e-4f
#define DRAY_EPSILON_64 1e-8f

#ifndef __CUDACC__
// make sure min / max resolve for both cuda and cpu
#include <algorithm>
#include <math.h>
#include <string.h> //resolve memcpy
using namespace std;
#endif

namespace dray
{
namespace detail
{

union Bits32 {
  float32 scalar;
  uint32 bits;
};

union Bits64 {
  float64 scalar;
  uint64 bits;
};

} // namespace detail

template <typename T> DRAY_EXEC T epsilon ()
{
  return 1;
}

template <> DRAY_EXEC float32 epsilon<float32> ()
{
  return DRAY_EPSILON_32;
}

template <> DRAY_EXEC float64 epsilon<float64> ()
{
  return DRAY_EPSILON_64;
}

DRAY_EXEC
float32 nan32 ()
{
  detail::Bits32 nan;
  nan.bits = DRAY_NAN_32;
  return nan.scalar;
}

DRAY_EXEC
float32 infinity32 ()
{
  detail::Bits32 inf;
  inf.bits = DRAY_INF_32;
  return inf.scalar;
}

DRAY_EXEC
float32 neg_infinity32 ()
{
  detail::Bits32 ninf;
  ninf.bits = DRAY_NG_INF_32;
  return ninf.scalar;
}

DRAY_EXEC
float64 nan64 ()
{
  detail::Bits64 nan;
  nan.bits = DRAY_NAN_64;
  return nan.scalar;
}

DRAY_EXEC
float64 infinity64 ()
{
  detail::Bits64 inf;
  inf.bits = DRAY_INF_64;
  return inf.scalar;
}

DRAY_EXEC
float64 neg_infinity64 ()
{
  detail::Bits64 ninf;
  ninf.bits = DRAY_NG_INF_64;
  return ninf.scalar;
}

template <typename T> DRAY_EXEC T infinity ();

template <> DRAY_EXEC float32 infinity<float32> ()
{
  return infinity32 ();
}

template <> DRAY_EXEC float64 infinity<float64> ()
{
  return infinity64 ();
}

template <typename T> DRAY_EXEC T nan();

template <> DRAY_EXEC float32 nan<float32> ()
{
  return nan32 ();
}

template <> DRAY_EXEC float64 nan<float64> ()
{
  return nan64 ();
}

template <typename T> DRAY_EXEC T neg_infinity ();

template <> DRAY_EXEC float32 neg_infinity<float32> ()
{
  return neg_infinity32 ();
}

template <> DRAY_EXEC float64 neg_infinity<float64> ()
{
  return neg_infinity64 ();
}

//
// count leading zeros
//
DRAY_EXEC
int32 clz (uint32 x)
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
  if (y != 0) return int32 (n - 2);
  return int32 (n - x);
}

DRAY_EXEC
float64 pi ()
{
  return 3.14159265358979323846264338327950288;
}

DRAY_EXEC
float32 rcp (float32 f)
{
  return 1.0f / f;
}

DRAY_EXEC
float64 rcp (float64 f)
{
  return 1.0 / f;
}

DRAY_EXEC
float64 rcp_safe (float64 f)
{
  return rcp ((fabs (f) < 1e-8) ? (signbit (f) ? -1e-8 : 1e-8) : f);
}

DRAY_EXEC
float32 rcp_safe (float32 f)
{
  return rcp ((fabs (f) < 1e-8f) ? (signbit (f) ? -1e-8f : 1e-8f) : f);
}

template <typename T>
DRAY_EXEC T clamp (const T &val, const T &min_val, const T &max_val)
{
  return min (max_val, max (min_val, val));
}

// Recursive integer power template, for nonnegative powers.
template <int32 b, int32 p> struct IntPow
{
  enum
  {
    val = IntPow<b, p / 2>::val * IntPow<b, p - p / 2>::val
  };
};

// Base cases.
template <int32 b> struct IntPow<b, 1>
{
  enum
  {
    val = b
  };
};
template <int32 b> struct IntPow<b, 0>
{
  enum
  {
    val = 1
  };
};

// Same thing but using a constexpr function.
/// constexpr int32 intPow(int32 b, uint32 p)
/// {
///   return (!p ? 1 : p == 1 ? b : intPow(b, p/2) * intPow(b, p-p/2));  // Good if the syntax tree could share leaves?
/// }
constexpr int32 intPow (int32 b, uint32 p, int32 a = 1)
{
  return (!p ? a : intPow (b, p - 1, a * b)); // Continuation, linear syntax tree.
}

static constexpr DRAY_EXEC float32 pi_180f ()
{
  return 0.01745329251994329547437168059786927f;
}
static constexpr DRAY_EXEC float64 pi_180 ()
{
  return 0.01745329251994329547437168059786927;
}


} // namespace dray
#endif

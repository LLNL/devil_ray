// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SIGN_VEC_HPP
#define DRAY_SIGN_VEC_HPP

#include <dray/exports.hpp>
#include <dray/math.hpp>
#include <dray/types.hpp>

namespace dray
{
  class SignVec
  {
    private:
      uint8 m_bits;

    public:
      DRAY_EXEC SignVec();
      DRAY_EXEC SignVec(int32 u);
      DRAY_EXEC SignVec(int32 u, int32 v);
      DRAY_EXEC SignVec(int32 u, int32 v, int32 w);
      DRAY_EXEC SignVec(int32 u, int32 v, int32 w, int32 s);
      DRAY_EXEC SignVec(const SignVec &) = default;
      DRAY_EXEC SignVec(SignVec &&) = default;
      DRAY_EXEC SignVec & operator=(const SignVec &) = default;
      DRAY_EXEC SignVec & operator=(SignVec &&) = default;

      DRAY_EXEC int32 at(int32 axis) const;
      DRAY_EXEC void at(int32 axis, int32 sign_val);

      DRAY_EXEC bool operator==(const SignVec &that) const;

      template <typename T, int32 S>
      DRAY_EXEC T combine(const Vec<T, S> &vectors) const;

      template <typename T, int32 S>
      DRAY_EXEC Vec<T, S> vec() const;

      DRAY_EXEC SignVec min(const SignVec &other) const;

      DRAY_EXEC SignVec inverse() const;

      DRAY_EXEC int8 dot(const SignVec &other) const;
  };

  inline std::ostream &operator<< (std::ostream &out, const SignVec &sign_vec)
  {
    for (int32 a = 0; a < 4; ++a)
      out << (sign_vec.at(a) < 0 ? '-' : sign_vec.at(a) > 0 ? '+' : '_');
    return out;
  }

}//namespace dray


namespace dray
{
  // Representation:
  //
  // 8 bits: {hgfedcba}
  // h,f,d,b = (is_negative)[3,2,1,0]
  // g,e,c,a = (magnitude)[3,2,1,0]

  DRAY_EXEC SignVec::SignVec()
    : m_bits(0) { } 

  DRAY_EXEC SignVec::SignVec(int32 u)
    : SignVec()
  {
    at(0, u);
  }

  DRAY_EXEC SignVec::SignVec(int32 u, int32 v)
    : SignVec()
  {
    at(0, u);
    at(1, v);
  }

  DRAY_EXEC SignVec::SignVec(int32 u, int32 v, int32 w)
    : SignVec()
  {
    at(0, u);
    at(1, v);
    at(2, w);
  }

  DRAY_EXEC SignVec::SignVec(int32 u, int32 v, int32 w, int32 s)
    : SignVec()
  {
    at(0, u);
    at(1, v);
    at(2, w);
    at(3, s);
  }

  DRAY_EXEC int32 SignVec::at(int32 axis) const
  {
    const uint8 bits = (m_bits >> (2*axis)) & 3u;
    const uint8 plus_2_unsigned = bits ^ 2u;
    const int32 plus_2_signed = plus_2_unsigned;
    return plus_2_signed - 2;
  }

  DRAY_EXEC void SignVec::at(int32 axis, int32 sign_val)
  {
    int32 sign_bits = (sign_val != 0);
    sign_bits |= (sign_val < 0) << 1;
    m_bits = (m_bits & (~(3u << (2*axis)))) | (sign_bits << (2*axis));
  }

    DRAY_EXEC bool SignVec::operator==(const SignVec &that) const
  {
    return m_bits == that.m_bits;
  }

  template <typename T, int32 S>
  DRAY_EXEC T SignVec::combine(const Vec<T, S> &vectors) const
  {
    T sum;
    sum = 0;
    for (int32 a = 0; a < S; ++a)
      sum += vectors[a] * at(a);
    return sum;
  }

  template <typename T, int32 S>
  DRAY_EXEC Vec<T, S> SignVec::vec() const
  {
    Vec<T, S> v;
    for (int32 a = 0; a < S; ++a)
      v[a] = at(a);
    return v;
  }

  DRAY_EXEC SignVec SignVec::min(const SignVec &other) const
  {
    constexpr uint8 select_signs = ( 128u  | 32u |  8u |  2u );
    constexpr uint8 select_mags =  (      64u | 16u |  4u |  1u );
    uint8 neg_ones = (m_bits | other.m_bits) & select_signs;
    neg_ones = neg_ones | (neg_ones >> 1);
    uint8 zeros_ones = (m_bits & other.m_bits) & select_mags;
    uint8 answer = neg_ones | zeros_ones;

    SignVec result;
    result.m_bits = answer;
    return result;
  }

  DRAY_EXEC SignVec SignVec::inverse() const
  {
    constexpr uint8 select_signs = ( 128u  | 32u |  8u |  2u );

    SignVec result;
    result.m_bits = m_bits ^ select_signs;

    return result;
  }

  DRAY_EXEC int8 SignVec::dot(const SignVec &other) const
  {
    constexpr uint8 select_signs = ( 128u  | 32u |  8u |  2u );
    constexpr uint8 select_mags =  (      64u | 16u |  4u |  1u );

    const uint8 &x = m_bits;
    const uint8 &y = other.m_bits;

    // Compute component-wise product vector.
    // -1 behaves like (and is represented as) 3 modulo 4.
    // Separate 1's place and 2's place, use distributive rule,
    // and ignore 4's place.
    const uint8 p =
        (((x&(y<<1)) | (y&(x<<1))) & select_signs)  // cross terms
      | ((x&y) & select_mags);                      // ones digits

    // Add the products as offset positive numbers
    // and then un-offset as signed.
    constexpr int32 D = 4;
    uint8 sum_plus_extra = 0;
    for (int32 d = 0; d < D; ++d)
      sum_plus_extra += ((uint8(p) >> (2*d)) & 3u) ^ 2u;
    int8 sum = int8(sum_plus_extra) - D * 2;
    return sum;
  }

}//namespace dray

#endif//DRAY_SIGN_VEC_HPP

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

      template <typename T, int32 S>
      DRAY_EXEC T combine(const Vec<T, S> &vectors) const;

      template <typename T, int32 S>
      DRAY_EXEC Vec<T, S> vec() const;
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
    int32 sign_bits = (m_bits >> (2*axis)) & 3u;
    int32 sign_val = sign_bits & 1u;
    if (sign_bits & 2u)
      sign_val = -sign_val;
    return sign_val;
  }

  DRAY_EXEC void SignVec::at(int32 axis, int32 sign_val)
  {
    int32 sign_bits = (sign_val != 0);
    sign_bits |= (sign_val < 0) << 1;
    m_bits = (m_bits & (~(3u << (2*axis)))) | (sign_bits << (2*axis));
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

}//namespace dray

#endif//DRAY_SIGN_VEC_HPP

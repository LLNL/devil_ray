// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FIXED_POINT_HPP
#define DRAY_FIXED_POINT_HPP

#include <dray/types.hpp>
#include <dray/exports.hpp>

#include <assert.h>
#include <iosfwd>


namespace dray
{
  namespace detail
  {
    // float64 -> uint64
    // float32 -> uint32
    template <typename FType> struct MatchPrecision { };
    template <> struct MatchPrecision<float64> {
      using IntType = uint64;
      static constexpr int32 bits = 64;
    };
    template <> struct MatchPrecision<float32> {
      using IntType = uint32;
      static constexpr int32 bits = 32;
    };
  }

  /** -----------------------------------
   * FixedPoint
   *  @brief Fixed-point arithmetic
   * -----------------------------------
   */
  class FixedPoint
  {
    public:
      using IntType = typename detail::MatchPrecision<Float>::IntType;
      static constexpr int32 bits = detail::MatchPrecision<Float>::bits;

      enum LargeFactor { TWO = 1 };
      enum SmallFactor { HALF = 1 };

    private:
      IntType m_uint;
      DRAY_EXEC FixedPoint(IntType u) : m_uint(u) {};

    public:
      DRAY_EXEC FixedPoint();
      DRAY_EXEC FixedPoint(Float f);
      DRAY_EXEC void operator=(Float f);

      DRAY_EXEC FixedPoint(const FixedPoint &) = default;
      DRAY_EXEC FixedPoint(FixedPoint &&) = default;
      DRAY_EXEC FixedPoint & operator=(const FixedPoint &) = default;
      DRAY_EXEC FixedPoint & operator=(FixedPoint &&) = default;

      DRAY_EXEC operator float32() const;
      DRAY_EXEC operator float64() const;

      DRAY_EXEC FixedPoint operator+(const FixedPoint &b) const;
      DRAY_EXEC FixedPoint operator-(const FixedPoint &b) const;
      DRAY_EXEC FixedPoint operator*(const LargeFactor &f) const;
      DRAY_EXEC FixedPoint operator*(const SmallFactor &f) const;
      DRAY_EXEC FixedPoint operator/(const LargeFactor &f) const;
      DRAY_EXEC FixedPoint operator/(const SmallFactor &f) const;
      DRAY_EXEC FixedPoint & operator+=(const FixedPoint &b);
      DRAY_EXEC FixedPoint & operator-=(const FixedPoint &b);
      DRAY_EXEC FixedPoint & operator*=(const LargeFactor &f);
      DRAY_EXEC FixedPoint & operator*=(const SmallFactor &f);
      DRAY_EXEC FixedPoint & operator/=(const LargeFactor &f);
      DRAY_EXEC FixedPoint & operator/=(const SmallFactor &f);

      DRAY_EXEC uint8 digit(int32 place) const;  // (0th, 1st) place are (1, 1/2).
      DRAY_EXEC void digit(int32 place, uint8 bit);

      friend std::ostream & operator<<(std::ostream &os, const FixedPoint &arg);
      DRAY_EXEC friend FixedPoint trunc(const FixedPoint &arg, int32 places);  // unchanged if places=bits.
  };
}


namespace dray
{

  // FixedPoint()
  DRAY_EXEC FixedPoint::FixedPoint()
  {
    m_uint = IntType(0u);
  }

  // FixedPoint()
  DRAY_EXEC FixedPoint::FixedPoint(Float f)
  {
    constexpr IntType maximum = IntType(1u) << (bits - 1);
    m_uint = f * maximum;
  }

  // operator=()
  DRAY_EXEC void FixedPoint::operator=(Float f)
  {
    constexpr IntType maximum = IntType(1u) << (bits - 1);
    m_uint = f * maximum;
  }

  // float32()
  DRAY_EXEC FixedPoint::operator float32() const
  {
    constexpr IntType maximum = IntType(1u) << (bits - 1);
    return float32(m_uint) / float32(maximum);
  }

  // float64()
  DRAY_EXEC FixedPoint::operator float64() const
  {
    constexpr IntType maximum = IntType(1u) << (bits - 1);
    return float64(m_uint) / float64(maximum);
  }

  // operator+()
  DRAY_EXEC FixedPoint FixedPoint::operator+(const FixedPoint &b) const
  {
    return FixedPoint(m_uint + b.m_uint);
  }

  // operator-()
  DRAY_EXEC FixedPoint FixedPoint::operator-(const FixedPoint &b) const
  {
    return FixedPoint(m_uint - b.m_uint);
  }

  // operator*()
  DRAY_EXEC FixedPoint FixedPoint::operator*(const LargeFactor &f) const
  {
    return FixedPoint(m_uint << f);
  }
  // operator*()
  DRAY_EXEC FixedPoint FixedPoint::operator*(const SmallFactor &f) const
  {
    return FixedPoint(m_uint >> f);
  }

  // operator/()
  DRAY_EXEC FixedPoint FixedPoint::operator/(const LargeFactor &f) const
  {
    return FixedPoint(m_uint >> f);
  }
  // operator/()
  DRAY_EXEC FixedPoint FixedPoint::operator/(const SmallFactor &f) const
  {
    return FixedPoint(m_uint << f);
  }

  // operator+=()
  DRAY_EXEC FixedPoint & FixedPoint::operator+=(const FixedPoint &b)
  {
    return operator=(operator+(b));
  }

  // operator-=()
  DRAY_EXEC FixedPoint & FixedPoint::operator-=(const FixedPoint &b)
  {
    return operator=(operator-(b));
  }

  // operator*=()
  DRAY_EXEC FixedPoint & FixedPoint::operator*=(const LargeFactor &f)
  {
    return operator=(operator*(f));
  }
  // operator*=()
  DRAY_EXEC FixedPoint & FixedPoint::operator*=(const SmallFactor &f)
  {
    return operator=(operator*(f));
  }

  // operator/=()
  DRAY_EXEC FixedPoint & FixedPoint::operator/=(const LargeFactor &f)
  {
    return operator=(operator/(f));
  }
  // operator/=()
  DRAY_EXEC FixedPoint & FixedPoint::operator/=(const SmallFactor &f)
  {
    return operator=(operator/(f));
  }

  // digit()
  DRAY_EXEC uint8 FixedPoint::digit(int32 place) const
  {
    constexpr IntType ones_place_mask = IntType(1u) << (bits - 1);
    const IntType place_mask = (ones_place_mask >> place);
    return bool(m_uint & place_mask);
  }

  // digit()
  DRAY_EXEC void FixedPoint::digit(int32 place, uint8 bit)
  {
    constexpr IntType ones_place_mask = IntType(1u) << (bits - 1);
    const IntType place_mask = (ones_place_mask >> place);
    const IntType background = IntType(0u) - bool(bit);
    m_uint = (m_uint & (~place_mask)) | (background & place_mask);
  }


  DRAY_EXEC FixedPoint trunc(const FixedPoint &arg, int32 places)
  {
    using IntType = FixedPoint::IntType;
    assert(places >= 0);
    const IntType all_bits = -IntType(1u);
    const IntType lower_bits = all_bits >> places;
    return FixedPoint(arg.m_uint & (~lower_bits));
  }
}

#endif//DRAY_FIXED_POINT_HPP

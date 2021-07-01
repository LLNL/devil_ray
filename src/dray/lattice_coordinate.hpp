// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LATTICE_COORDINATE_HPP
#define DRAY_LATTICE_COORDINATE_HPP

#include <dray/types.hpp>
#include <dray/exports.hpp>
#include <dray/fixed_point.hpp>
#include <dray/vec.hpp>

#include <assert.h>
#include <iosfwd>


namespace dray
{
  template <int32 S>
  class LatticeCoord
  {
    private:
      FixedPoint m_vec[S];

    public:
      DRAY_EXEC LatticeCoord();

      DRAY_EXEC LatticeCoord(const FixedPoint *coords);

      DRAY_EXEC LatticeCoord(const LatticeCoord &) = default;
      DRAY_EXEC LatticeCoord(LatticeCoord &&) = default;
      DRAY_EXEC LatticeCoord & operator=(const LatticeCoord &) = default;
      DRAY_EXEC LatticeCoord & operator=(LatticeCoord &&) = default;

      DRAY_EXEC const FixedPoint * coords() const;
      DRAY_EXEC const FixedPoint & at(int32 d) const;
      DRAY_EXEC void set(int32 d, const FixedPoint &fp);

      template <typename T>
      DRAY_EXEC Vec<T, S> vec() const;
  };

}


namespace dray
{
  // LatticeCoord()
  template <int32 S>
  DRAY_EXEC LatticeCoord<S>::LatticeCoord()
  {
    for (int32 d = 0; d < S; ++d)
      m_vec[d] = 0.0f;
  }

  // LatticeCoord()
  template <int32 S>
  DRAY_EXEC LatticeCoord<S>::LatticeCoord(const FixedPoint *coords)
  {
    for (int32 d = 0; d < S; ++d)
      m_vec[d] = coords[d];
  }

  // coords()
  template <int32 S>
  DRAY_EXEC const FixedPoint * LatticeCoord<S>::coords() const
  {
    return m_vec;
  }

  // at()
  template <int32 S>
  DRAY_EXEC const FixedPoint & LatticeCoord<S>::at(int32 d) const
  {
    return m_vec[d];
  }

  // set()
  template <int32 S>
  DRAY_EXEC void LatticeCoord<S>::set(int32 d, const FixedPoint &fp)
  {
    m_vec[d] = fp;
  }

  // vec()
  template <int32 S>
  template <typename T>
  DRAY_EXEC Vec<T, S> LatticeCoord<S>::vec() const
  {
    Vec<T, S> result;
    for (int32 d = 0; d < S; ++d)
      result[d] = T(m_vec[d]);
    return result;
  }
}

#endif//DRAY_LATTICE_COORDINATE_HPP

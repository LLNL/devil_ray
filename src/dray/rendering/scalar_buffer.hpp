// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SCALAR_BUFFER_HPP
#define DRAY_SCALAR_BUFFER_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{


class ScalarBuffer
{
  protected:
  Array<Float> m_scalars;
  Array<float32> m_depths;
  int32 m_width;
  int32 m_height;
  Float m_clear_value;

  public:
  ScalarBuffer ();
  ScalarBuffer (const int32 width, const int32 height);

  int32 width () const;
  int32 height () const;

  void clear (); // clear out the scalar buffer with clear value, set depths to inf
  //void save (const std::string name);
  //void save_depth (const std::string name);

  friend class DeviceScalarBuffer;
};

} // namespace dray
#endif

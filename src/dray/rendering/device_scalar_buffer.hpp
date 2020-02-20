// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_SCALAR_BUFFER_HPP
#define DRAY_DEVICE_SCALAR_BUFFER_HPP

#include <dray/rendering/scalar_buffer.hpp>

namespace dray
{

struct DeviceScalarBuffer
{
  Float *m_scalars;
  float32 *m_depths;

  DeviceScalarBuffer () = delete;

  DeviceScalarBuffer (ScalarBuffer &scalar_buffer)
  {
    m_scalars = scalar_buffer.m_scalars.get_device_ptr ();
    m_depths = scalar_buffer.m_depths.get_device_ptr ();
  }

  void DRAY_EXEC set_scalar(const int32 &index, const Float &scalar)
  {
    m_scalars[index] = scalar;
  }

  void DRAY_EXEC set_depth (const int32 &index, const float32 &depth)
  {
    m_depths[index] = depth;
  }
};

} // namespace dray
#endif

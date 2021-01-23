// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_ENV_MAP_HPP
#define DRAY_DEVICE_ENV_MAP_HPP

#include <dray/rendering/env_map.hpp>
#include <dray/error.hpp>

namespace dray
{

struct DeviceEnvMap
{
  const Vec<float32, 3> *m_colors;
  const int32 m_width;
  const int32 m_height;
  float32 m_scale;

  DeviceEnvMap() = delete;

  DeviceEnvMap (EnvMap &map)
   : m_colors(map.m_image.get_device_ptr()),
     m_width(map.m_width),
     m_height(map.m_height),
     m_scale(1.f)
  {
  }

  void scale(const float32 scale)
  {
    m_scale = scale;
  }

  // needs to be a normalized direction
  Vec<float32,3> DRAY_EXEC color (const Vec<float32,3> &dir) const
  {
    float32 x = (atan2(dir[2],dir[0]) + dray::pi()) / (dray::pi() * 2.f);
    float32 y = acos(dir[1]) / dray::pi();
    int32 xi = float32(m_width-1) * x;
    int32 yi = float32(m_height-1) * y;
    int32 index = yi * m_width + xi;
    index = clamp(index, 0, m_width * m_height - 1);

    return m_colors[index] * m_scale;
  }

  float32 pdf() const
  {
    return 1.f / 4.f * pi();
  }
};

} // namespace dray
#endif

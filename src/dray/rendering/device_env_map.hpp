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
  float32 m_radius;
  const DeviceDistribution2D m_distribution;
  const Vec<int32, 3> m_up_map;

  DeviceEnvMap() = delete;

  DeviceEnvMap (EnvMap &map)
   : m_colors(map.m_image.get_device_ptr()),
     m_width(map.m_width),
     m_height(map.m_height),
     m_scale(1.f),
     m_distribution(map.m_distribution),
     m_radius(map.m_radius),
     m_up_map({{0,2,1}})
  {
  }

  void scale(const float32 scale)
  {
    m_scale = scale;
  }

  DRAY_EXEC
  Vec<float32,3> sample(const Vec<float32,2> &rand, float32 &pdf) const
  {
    Vec<float32,2> u = m_distribution.sample(rand, pdf);
    float32 theta = u[1] * pi();
    float32 phi = u[0] * pi() * 2.f;
    float32 sin_phi = sin(phi);
    float32 cos_phi = cos(phi);
    float32 cos_theta = cos(theta);
    float32 sin_theta = sin(theta);

    Vec<float32,3> dir = {{sin_theta * cos_phi,
                           sin_theta * sin_phi,
                           cos_theta}};
    if(sin_theta == 0)
    {
      pdf = 0;
    }
    else
    {
      // tranform pdf in terms of solid angle
      pdf = pdf / (2.f * pi() * pi()  * sin_theta);
    }
    dir.normalize();
    return {{dir[m_up_map[0]], dir[m_up_map[1]], dir[m_up_map[2]]}};
  }

  DRAY_EXEC
  Vec<float32,2> samplei(const Vec<float32,2> &rand) const
  {
    float32 pdf;
    Vec<float32,2> u = m_distribution.sample(rand, pdf);
    return u;
  }

  // needs to be a normalized direction
  Vec<float32,3> DRAY_EXEC color (const Vec<float32,3> &dir) const
  {

    const Vec<float32,3> mdir = {{dir[m_up_map[0]], dir[m_up_map[1]], dir[m_up_map[2]]}};
    // get the textel we can see what we are sampling
    float32 p = atan2(mdir[1],mdir[0]);
    if(p < 0.f) p = p + 2.f * pi();
    float32 x =  p / (pi() * 2.f);
    float32 y = acos(dray::clamp(mdir[2],-1.f, 1.f)) / pi();
    int32 xi = float32(m_width-1) * x;
    int32 yi = float32(m_height-1) * y;
    int32 index = yi * m_width + xi;
    index = clamp(index, 0, m_width * m_height - 1);

    return m_colors[index] * m_scale;
  }


  DRAY_EXEC
  float32 pdf() const
  {
    return 1.f / (pi() * m_radius * m_radius);
  }
};

} // namespace dray
#endif

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SPHERE_LIGHT_HPP
#define DRAY_SPHERE_LIGHT_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>

#include <vector>

namespace dray
{

enum LightType
{
  sphere = 0,
  quad = 1
};

struct SphereLight
{
  Vec<float32, 3> m_pos = {{0.f, 0.f, 0.f}};
  float32         m_radius;
  Vec<float32, 3> m_intensity = {{1.f, 1.f, 1.0f}};
  // 7 floats
};

struct QuadLight
{
  Vec<float32, 3> m_v0 = {{0.f, 0.f, 0.f}};
  Vec<float32, 3> m_v1 = {{1.f, 0.f, 0.f}};
  Vec<float32, 3> m_v2 = {{0.f, 1.f, 0.f}};
  Vec<float32, 3> m_v3 = {{1.f, 1.f, 0.f}};
  Vec<float32, 3> m_intensity = {{1.f, 1.f, 1.0f}};
  // 15 floats
};

struct LightContainer
{
  Array<float32> m_data;
  Array<int32> m_offsets;
  Array<int32> m_types;
  int32 m_num_lights;
  void pack(const std::vector<SphereLight> &sphere_lights,
            const std::vector<QuadLight> &quad_lights)
  {
    int32 size = 0;
    int32 raw_size = 0;

    size += sphere_lights.size();
    size += quad_lights.size();
    m_num_lights = size;

    raw_size += 7 * sphere_lights.size();
    raw_size += 15 * quad_lights.size();

    m_data.resize(raw_size);
    m_offsets.resize(size);
    m_types.resize(size);

    float32 *raw_ptr = m_data.get_host_ptr();
    int32 *offset_ptr = m_offsets.get_host_ptr();
    int32 *type_ptr = m_types.get_host_ptr();

    int32 current_offset = 0;
    int32 current_light = 0;
    for(int i = 0; i < sphere_lights.size(); ++i)
    {
      SphereLight light = sphere_lights[i];
      raw_ptr[current_offset + 0] = light.m_pos[0];
      raw_ptr[current_offset + 1] = light.m_pos[1];
      raw_ptr[current_offset + 2] = light.m_pos[2];
      raw_ptr[current_offset + 3] = light.m_radius;
      raw_ptr[current_offset + 4] = light.m_intensity[0];
      raw_ptr[current_offset + 5] = light.m_intensity[1];
      raw_ptr[current_offset + 6] = light.m_intensity[2];
      offset_ptr[current_light] = current_offset;
      current_offset += 7;
      type_ptr[current_light] = LightType::sphere;
      current_light++;
    }

    for(int i = 0; i < quad_lights.size(); ++i)
    {
      QuadLight light = quad_lights[i];
      raw_ptr[current_offset + 0] = light.m_v0[0];
      raw_ptr[current_offset + 1] = light.m_v0[1];
      raw_ptr[current_offset + 2] = light.m_v0[2];
      raw_ptr[current_offset + 3] = light.m_v1[0];
      raw_ptr[current_offset + 4] = light.m_v1[1];
      raw_ptr[current_offset + 5] = light.m_v1[2];
      raw_ptr[current_offset + 6] = light.m_v2[0];
      raw_ptr[current_offset + 7] = light.m_v2[1];
      raw_ptr[current_offset + 8] = light.m_v2[2];
      raw_ptr[current_offset + 9]  = light.m_v3[0];
      raw_ptr[current_offset + 10] = light.m_v3[1];
      raw_ptr[current_offset + 11] = light.m_v3[2];
      raw_ptr[current_offset + 12] = light.m_intensity[0];
      raw_ptr[current_offset + 13] = light.m_intensity[1];
      raw_ptr[current_offset + 14] = light.m_intensity[2];
      offset_ptr[current_light] = current_offset;
      current_offset += 15;
      type_ptr[current_light] = LightType::quad;
      current_light++;
    }
  }
};

struct DeviceLightContainer
{
  const float32 *m_data;
  const int32 *m_offsets;
  const int32 *m_types;
  const int32 m_num_lights;
  DeviceLightContainer(LightContainer &lights)
    : m_data(lights.m_data.get_device_ptr_const()),
      m_offsets(lights.m_offsets.get_device_ptr_const()),
      m_types(lights.m_offsets.get_device_ptr_const()),
      m_num_lights(lights.m_num_lights)
  {
  }
};

std::ostream &operator<< (std::ostream &out, const SphereLight &light);
std::ostream &operator<< (std::ostream &out, const QuadLight &light);

} // namespace dray
#endif

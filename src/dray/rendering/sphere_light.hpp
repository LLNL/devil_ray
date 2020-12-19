// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SPHERE_LIGHT_HPP
#define DRAY_SPHERE_LIGHT_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/error.hpp>
#include <dray/rendering/sampling.hpp>

#include <vector>

namespace dray
{

enum LightType
{
  sphere = 0,
  tri = 1
};

struct SphereLight
{
  Vec<float32, 3> m_pos = {{0.f, 0.f, 0.f}};
  float32         m_radius;
  Vec<float32, 3> m_intensity = {{1.f, 1.f, 1.0f}};
  // 7 floats

  DRAY_EXEC Vec<float32,3>
  sample(const Vec<float32,3> &point,
         const Vec<float32,2> &u, // random
         float32 &pdf,
         bool debug = false) const
  {
    return sphere_sample(m_pos,m_radius,point,u,pdf,debug);
  }
};

struct TriangleLight
{
  Vec<float32, 3> m_v0 = {{0.f, 0.f, 0.f}};
  Vec<float32, 3> m_v1 = {{1.f, 0.f, 0.f}};
  Vec<float32, 3> m_v2 = {{0.f, 1.f, 0.f}};
  Vec<float32, 3> m_intensity = {{1.f, 1.f, 1.0f}};

  DRAY_EXEC Vec<float32,3>
  sample(const Vec<float32,3> &point,
         const Vec<float32,2> &u, // random
         float32 &pdf,
         bool debug = false) const
  {
    // uniform triangle sampling
    float32 su = sqrt(u[0]);
    // barycentric coords
    float32 bu = 1.f - su;
    float32 bv = su * u[1];
    float32 bn = 1.f - bu - bv;

    pdf = 1.f / (0.5f *cross(m_v1 - m_v0, m_v2 - m_v0).magnitude());
    if(debug)
    {
      std::cout<<"[tri light] PDF "<<pdf<<"\n";
      std::cout<<"[tri light] area "<<1.f / pdf<<"\n";
    }
    return m_v0 * bu + m_v1 * bv + m_v2 * bn;
  }
};

struct LightContainer
{
  Array<float32> m_data;
  Array<Vec<float32,3>> m_intensities;
  Array<int32> m_offsets;
  Array<int32> m_types;
  int32 m_num_lights;

  SphereLight sphere_light(int32 idx)
  {
    if(idx >= m_num_lights)
    {
      DRAY_ERROR("Invalid light idx "<<idx);
    }
    if(m_types.get_value(idx) != LightType::sphere)
    {
      DRAY_ERROR("Light idx is not a sphere");
    }
    SphereLight light;
    int32 offset = m_offsets.get_value(idx);
    light.m_pos[0] = m_data.get_value(offset + 0);
    light.m_pos[1] = m_data.get_value(offset + 1);
    light.m_pos[2] = m_data.get_value(offset + 2);
    light.m_radius = m_data.get_value(offset + 3);
    light.m_intensity = m_intensities.get_value(idx);
    return light;
  }

  TriangleLight triangle_light(int32 idx)
  {
    if(idx >= m_num_lights)
    {
      DRAY_ERROR("Invalid light idx "<<idx);
    }
    if(m_types.get_value(idx) != LightType::tri)
    {
      DRAY_ERROR("Light idx is not a triangle");
    }

    TriangleLight light;
    int32 offset = m_offsets.get_value(idx);
    light.m_v0[0] = m_data.get_value(offset + 0);
    light.m_v0[1] = m_data.get_value(offset + 1);
    light.m_v0[2] = m_data.get_value(offset + 2);
    light.m_v1[0] = m_data.get_value(offset + 3);
    light.m_v1[1] = m_data.get_value(offset + 4);
    light.m_v1[2] = m_data.get_value(offset + 5);
    light.m_v2[0] = m_data.get_value(offset + 6);
    light.m_v2[1] = m_data.get_value(offset + 7);
    light.m_v2[2] = m_data.get_value(offset + 8);

    light.m_intensity = m_intensities.get_value(idx);
    return light;
  }

  Vec<float32,3> intensity(const int32 idx)
  {
    if(idx >= m_num_lights)
    {
      DRAY_ERROR("Invalid light idx "<<idx);
    }
    return m_intensities.get_value(idx);
  }

  void pack(const std::vector<SphereLight> &sphere_lights,
            const std::vector<TriangleLight> &tri_lights)
  {
    int32 size = 0;
    int32 raw_size = 0;

    size += sphere_lights.size();
    size += tri_lights.size();
    m_num_lights = size;

    raw_size += 4 * sphere_lights.size();
    raw_size += 9 * tri_lights.size();

    m_data.resize(raw_size);
    m_intensities.resize(size);
    m_offsets.resize(size);
    m_types.resize(size);

    float32 *raw_ptr = m_data.get_host_ptr();
    Vec<float32,3> *intensities_ptr = m_intensities.get_host_ptr();
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

      intensities_ptr[current_light] = light.m_intensity;
      offset_ptr[current_light] = current_offset;
      current_offset += 4;
      type_ptr[current_light] = LightType::sphere;
      current_light++;
    }

    for(int i = 0; i < tri_lights.size(); ++i)
    {
      TriangleLight light = tri_lights[i];
      raw_ptr[current_offset + 0] = light.m_v0[0];
      raw_ptr[current_offset + 1] = light.m_v0[1];
      raw_ptr[current_offset + 2] = light.m_v0[2];
      raw_ptr[current_offset + 3] = light.m_v1[0];
      raw_ptr[current_offset + 4] = light.m_v1[1];
      raw_ptr[current_offset + 5] = light.m_v1[2];
      raw_ptr[current_offset + 6] = light.m_v2[0];
      raw_ptr[current_offset + 7] = light.m_v2[1];
      raw_ptr[current_offset + 8] = light.m_v2[2];
      intensities_ptr[current_light] = light.m_intensity;
      offset_ptr[current_light] = current_offset;
      current_offset += 9;
      type_ptr[current_light] = LightType::tri;
      current_light++;
    }
  }
};

struct DeviceLightContainer
{
  const float32 *m_data;
  const Vec<float32,3> *m_intensities;
  const int32 *m_offsets;
  const int32 *m_types;
  const int32 m_num_lights;

  DeviceLightContainer(LightContainer &lights)
    : m_data(lights.m_data.get_device_ptr_const()),
      m_intensities(lights.m_intensities.get_device_ptr_const()),
      m_offsets(lights.m_offsets.get_device_ptr_const()),
      m_types(lights.m_types.get_device_ptr_const()),
      m_num_lights(lights.m_num_lights)
  {
  }

  DRAY_EXEC
  SphereLight sphere_light(int32 idx) const
  {
    if(idx >= m_num_lights)
    {
      DRAY_ERROR("Invalid light idx "<<idx);
    }
    if(m_types[idx] != LightType::sphere)
    {
      DRAY_ERROR("Light idx is not a sphere");
    }
    SphereLight light;
    int32 offset = m_offsets[idx];
    light.m_pos[0] = m_data[offset + 0];
    light.m_pos[1] = m_data[offset + 1];
    light.m_pos[2] = m_data[offset + 2];
    light.m_radius = m_data[offset + 3];
    light.m_intensity = m_intensities[idx];
    return light;
  }

  DRAY_EXEC
  TriangleLight triangle_light(int32 idx) const
  {
    TriangleLight light;
    int32 offset = m_offsets[idx];
    light.m_v0[0] = m_data[offset + 0];
    light.m_v0[1] = m_data[offset + 1];
    light.m_v0[2] = m_data[offset + 2];
    light.m_v1[0] = m_data[offset + 3];
    light.m_v1[1] = m_data[offset + 4];
    light.m_v1[2] = m_data[offset + 5];
    light.m_v2[0] = m_data[offset + 6];
    light.m_v2[1] = m_data[offset + 7];
    light.m_v2[2] = m_data[offset + 8];

    light.m_intensity = m_intensities[idx];
    return light;
  }

  DRAY_EXEC
  Vec<float32,3> intensity(int32 idx) const
  {
    if(idx >= m_num_lights)
    {
      DRAY_ERROR("Invalid light idx "<<idx);
    }
    return m_intensities[idx];
  }
};

std::ostream &operator<< (std::ostream &out, const SphereLight &light);
std::ostream &operator<< (std::ostream &out, const TriangleLight &light);

} // namespace dray
#endif

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/env_map.hpp>
#include <dray/io/hdr_image_reader.hpp>

namespace dray
{

namespace detail
{

Array<float32>
scalar_image(const Array<Vec<float32,3>> image,
             const int32 width,
             const int32 height,
             float32 &ave)
{
  // set up the importance sampling
  Array<float32> intensity;
  const int32 size = width * height;
  intensity.resize(size);
  float32 * i_ptr = intensity.get_host_ptr();
  const Vec<float32,3> * rgb_ptr = image.get_host_ptr_const();

  float32 sum = 0.f;
  for(int i = 0; i < size; ++i)
  {
    Vec<float32,3> rgb = rgb_ptr[i];
    float32 value = 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
    sum += value;

    // appoximate the solid angle and weight
    float32 y = float32(i / width);
    float32 sin_theta = sin(pi() * (y + 0.5) / float32(height));
    value *= sin_theta;
    i_ptr[i] = value;
  }
  ave = sum / float32(size);
  return intensity;
}

} // namespace detail

EnvMap::EnvMap()
  : m_width(0),
    m_height(0),
    m_ave_radiance(0),
    m_radius(1)
{
  constant_color({{0.f, 0.f, 0.f}});
}

void
EnvMap::load(const std::string file_name)
{
  m_image = read_hdr_image(file_name, m_width, m_height);
  Array<float32> intensity = detail::scalar_image(m_image, m_width, m_height, m_ave_radiance);
  m_distribution = Distribution2D(intensity, m_width, m_height);
}

void
EnvMap::radius(const float32 radius)
{
  m_radius = radius;
}

float32
EnvMap::power()
{
  return m_radius * m_radius * pi() * m_ave_radiance;
}

void
EnvMap::image(Array<Vec<float32,3>> image, const int32 width, const int32 height)
{
  if(image.size() != width * height)
  {
    DRAY_ERROR("Bad image dimensions");
  }
  m_image = image;
  m_width = width;
  m_height = height;

  Array<float32> intensity = detail::scalar_image(m_image,width, height, m_ave_radiance);
  m_distribution = Distribution2D(intensity, m_width, m_height);
}

void
EnvMap::constant_color(const Vec<float32,3> color)
{
  m_width = 1;
  m_height = 1;
  m_image.resize(1);
  Vec<float32,3> *color_ptr = m_image.get_host_ptr();
  color_ptr[0] = color[0];
  color_ptr[1] = color[1];
  color_ptr[2] = color[2];
  m_ave_radiance = 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
}

} // namespace dray

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/env_map.hpp>
#include <dray/io/hdr_image_reader.hpp>

namespace dray
{

EnvMap::EnvMap()
  : m_width(0),
    m_height(0)
{
  constant_color({{0.f, 0.f, 0.f}});
}

void
EnvMap::load(const std::string file_name)
{
  m_image = read_hdr_image(file_name, m_width, m_height);
}

void
EnvMap::constant_color(const Vec<float32,3> color)
{
  m_width = 1;
  m_height = 1;
  m_image.resize(1);
  Vec<float32,3> *color_ptr = m_image.get_host_ptr();
  color_ptr[0] = 0;
}

} // namespace dray

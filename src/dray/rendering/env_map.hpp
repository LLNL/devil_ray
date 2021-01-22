// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ENV_MAP_HPP
#define DRAY_ENV_MAP_HPP

#include <dray/array.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class EnvMap
{
  int32 m_width;
  int32 m_height;
  Array<Vec<float32,3>> m_image;
public:
  EnvMap();
  void load(const std::string file_name);
  void constant_color(const Vec<float32,3> color);

  friend struct DeviceEnvMap;
};

} // namespace dray
#endif

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_PATH_DATA_HPP
#define DRAY_PATH_DATA_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

struct RayData
{
  Vec<float32,3> m_throughput;
  float32 m_brdf; // used for mixing pdfs
  int32 m_flags; // shadow ray, whas the last bounce specular..
};

struct Sample
{
  Vec<float32,4> m_color;
  Vec<float32,3> m_normal;
  float32 m_distance;
  int32 m_hit_flag;
};

struct Material
{
  Vec3f m_emmisive = {{0.f, 0.f, 0.f}};;
  float32 m_roughness = 0.25f;  // specular roughness
  float32 m_diff_ratio = 0.50f; // chance of being specular or diff
};

std::ostream &operator<< (std::ostream &out, const RayData &data);
std::ostream &operator<< (std::ostream &out, const Sample &sample);
std::ostream &operator<< (std::ostream &out, const Material &mat);

} // namespace dray
#endif

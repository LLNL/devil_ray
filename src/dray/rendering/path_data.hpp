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

enum RayFlags
{
  EMPTY = 0,
  SPECULAR = 1 << 0,
  DIFFUSE = 1 << 1,
  TRANSMITTANCE = 1 << 2,
  // rays that miss of are culled because of low contrib
  TERMINATE = 1 << 3,
  // invalid samples need to go
  INVALID = 1 << 4,
};

struct RayData
{
  Vec<float32,3> m_throughput;
  float32 m_pdf; // used for mixing pdfs
  int32 m_depth;
  RayFlags m_flags;
};

struct Sample
{
  Vec<float32,4> m_color;
  Vec<float32,3> m_normal;
  float32 m_distance;
  int32 m_hit_flag;
  int32 m_mat_id;
};

struct Material
{
  Vec3f m_emmisive = {{0.f, 0.f, 0.f}};
  float32 m_roughness = 0.25f;  // specular roughness
  float32 m_spec_trans = 0;
  float32 m_metallic = 0;
  float32 m_specular = 0.5;
  float32 m_anisotropic = 0;
  float32 m_subsurface = 0;
  float32 m_sheen_tint = 0;
  float32 m_spec_tint = 0;
  float32 m_clearcoat_gloss = 0;
  float32 m_clearcoat = 0;
  float32 m_sheen = 0;
  float32 m_ior = 1;
};

std::ostream &operator<< (std::ostream &out, const RayData &data);
std::ostream &operator<< (std::ostream &out, const Sample &sample);
std::ostream &operator<< (std::ostream &out, const Material &mat);

} // namespace dray
#endif

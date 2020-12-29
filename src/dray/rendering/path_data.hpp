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
  float32 m_pdf; // used for mixing pdfs
  int32 m_depth;
  bool m_is_specular;

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
  // TODO: change to metalic = 1-diff
  float32 m_diff_ratio = 0.50f; // chance of being specular or diff
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

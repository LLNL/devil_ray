// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SPHERE_LIGHT_HPP
#define DRAY_SPHERE_LIGHT_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

struct SphereLight
{
  Vec<float32, 3> m_pos = {{0.f, 0.f, 0.f}};
  float32         m_radius;
  Vec<float32, 3> m_intensity = {{1.f, 1.f, 1.0f}};
};

std::ostream &operator<< (std::ostream &out, const SphereLight &light);

} // namespace dray
#endif

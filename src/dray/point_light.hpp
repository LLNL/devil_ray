// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POINT_LIGHT_HPP
#define DRAY_POINT_LIGHT_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

struct PointLight
{
  Vec<float32, 3> m_pos = {0.f, 0.f, 0.f};
  Vec<float32, 3> m_amb = {0.2f, 0.2f, 0.2f};
  Vec<float32, 3> m_diff = {0.5f, 0.5f, 0.5f};
  Vec<float32, 3> m_spec = {0.8f, 0.8f, 0.8f};
  float32 m_spec_pow = 60.f;
};

static std::ostream &operator<< (std::ostream &out, const PointLight &light)
{
  out << "{"<<light.m_pos<<" "<<light.m_amb<<" "<<light.m_diff<<" ";
  out << light.m_spec<<" "<<light.m_spec_pow<<"} ";
  return out;
}

} // namespace dray
#endif

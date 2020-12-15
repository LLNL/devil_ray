// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/sphere_light.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const SphereLight &light)
{
  out << "{"<<light.m_pos<<" "<<light.m_radius<<" "<<light.m_intensity<<"}";
  return out;
}

std::ostream &operator<< (std::ostream &out, const TriangleLight &light)
{
  out << "{"<<light.m_v0<<" "<<light.m_v1
      <<" "<< light.m_v2<<" "<<" "<<light.m_intensity<<"}";
  return out;
}

} // namespace dray

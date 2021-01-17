// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/path_data.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const RayData &data)
{
  out << "{"<<data.m_throughput<<" "<<data.m_pdf<<" "<<data.m_is_specular<<"}";
  return out;
}

std::ostream &operator<< (std::ostream &out, const Material &mat)
{
  out << "{"<<mat.m_emmisive<<" "<<mat.m_roughness<<" "<<mat.m_metallic<<"}";
  return out;
}

std::ostream &operator<< (std::ostream &out, const Sample &sample)
{
  out << "{"<<sample.m_color<<" "<<sample.m_normal
      <<" "<<sample.m_distance<<" "<<" "<<sample.m_hit_flag<<"}";
  return out;
}

} // namespace dray

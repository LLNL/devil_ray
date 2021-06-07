// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/face_location.hpp>

namespace dray
{
  std::ostream &operator<< (std::ostream &out, const FaceLocation &loc)
  {
    out << loc.m_loc << ":" << loc.m_tangents;
    return out;
  }

  std::ostream &operator<< (std::ostream &out, const FaceTangents &face_tangents)
  {
    out << "(" << face_tangents.m_t[0]
        << "," << face_tangents.m_t[1] << ")";
    return out;
  }
}

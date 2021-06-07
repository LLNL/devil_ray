// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FACE_LOCATION_HPP
#define DRAY_FACE_LOCATION_HPP

#include <dray/location.hpp>
#include <dray/sign_vec.hpp>

namespace dray
{
  struct FaceTangents
  {
    SignVec m_t[2];
  };

  /**
   * 3D Location (cell and reference coordinate) + face discriminator.
   * This approach contrasts with the creation of face elements.
   */
  class FaceLocation
  {
    public:
      Location m_loc;
      FaceTangents m_tangents;

      enum HexFaceTangent { XY, XZ, YZ };
      enum TetFaceTangent { XYZ, XYW, XZW, YZW };

      DRAY_EXEC Location loc() const;
      DRAY_EXEC FaceTangents tangents() const;

      DRAY_EXEC static FaceTangents cube_face(HexFaceTangent tangent_id);
      DRAY_EXEC static FaceTangents tet_face(TetFaceTangent tangent_id);
  };
  std::ostream &operator<< (std::ostream &out, const FaceLocation &loc);
  std::ostream &operator<< (std::ostream &out, const FaceTangents &face_tangents);

  DRAY_EXEC Location FaceLocation::loc() const
  {
    return m_loc;
  }

  DRAY_EXEC FaceTangents FaceLocation::tangents() const
  {
    return m_tangents;
  }

  DRAY_EXEC FaceTangents FaceLocation::cube_face(HexFaceTangent tangent_id)
  {
    switch (tangent_id)
    {
      case XY: return {{SignVec(1,0,0), SignVec(0,1,0)}};
      case XZ: return {{SignVec(1,0,0), SignVec(0,0,1)}};
      case YZ: return {{SignVec(0,1,0), SignVec(0,0,1)}};
      default: return {};
    }
  }

  DRAY_EXEC FaceTangents FaceLocation::tet_face(TetFaceTangent tangent_id)
  {
    switch (tangent_id)
    {
      case XYZ: return {{SignVec(1,0,-1), SignVec(0,1,-1)}};
      case XYW: return {{SignVec(1,0,0), SignVec(0,1,0)}};
      case XZW: return {{SignVec(1,0,0), SignVec(0,0,1)}};
      case YZW: return {{SignVec(0,1,0), SignVec(0,0,1)}};
      default: return {};
    }
  }

}//namespace dray

#endif//DRAY_FACE_LOCATION_HPP

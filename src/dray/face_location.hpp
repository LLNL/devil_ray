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

    enum HexFaceTangent { XY, XZ, YZ };
    enum TetFaceTangent { XYZ, XYW, XZW, YZW };

    DRAY_EXEC static FaceTangents cube_face(HexFaceTangent tangent_id);
    DRAY_EXEC static FaceTangents tet_face(TetFaceTangent tangent_id);

    DRAY_EXEC static FaceTangents cube_face_xy();
    DRAY_EXEC static FaceTangents cube_face_xz();
    DRAY_EXEC static FaceTangents cube_face_yz();

    DRAY_EXEC static FaceTangents tet_face_xyz();
    DRAY_EXEC static FaceTangents tet_face_xyw();
    DRAY_EXEC static FaceTangents tet_face_xzw();
    DRAY_EXEC static FaceTangents tet_face_yzw();

    DRAY_EXEC bool operator==(const FaceTangents &that) const;
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

      DRAY_EXEC Location loc() const;
      DRAY_EXEC FaceTangents tangents() const;

      DRAY_EXEC void world_tangents(
          const Vec<Vec<Float, 3>, 3> &jacobian,
          Vec<Float, 3> &t0,
          Vec<Float, 3> &t1) const;
      DRAY_EXEC Vec<Float, 3> world_normal(
          const Vec<Vec<Float, 3>, 3> &jacobian) const;
  };
  std::ostream &operator<< (std::ostream &out, const FaceLocation &loc);
  std::ostream &operator<< (std::ostream &out, const FaceTangents &face_tangents);


  // FaceTangents::operator==()
  DRAY_EXEC bool FaceTangents::operator==(const FaceTangents &that) const
  {
    return m_t[0] == that.m_t[0] && m_t[1] == that.m_t[1];
  }

  // cube_face / tet_face convenience creators
  DRAY_EXEC FaceTangents FaceTangents::cube_face_xy() { return cube_face(XY); }
  DRAY_EXEC FaceTangents FaceTangents::cube_face_xz() { return cube_face(XZ); }
  DRAY_EXEC FaceTangents FaceTangents::cube_face_yz() { return cube_face(YZ); }
  DRAY_EXEC FaceTangents FaceTangents::tet_face_xyz() { return tet_face(XYZ); }
  DRAY_EXEC FaceTangents FaceTangents::tet_face_xyw() { return tet_face(XYW); }
  DRAY_EXEC FaceTangents FaceTangents::tet_face_xzw() { return tet_face(XZW); }
  DRAY_EXEC FaceTangents FaceTangents::tet_face_yzw() { return tet_face(YZW); }

  // cube_face()
  DRAY_EXEC FaceTangents FaceTangents::cube_face(HexFaceTangent tangent_id)
  {
    switch (tangent_id)
    {
      case XY: return {{SignVec(1,0,0), SignVec(0,1,0)}};
      case XZ: return {{SignVec(1,0,0), SignVec(0,0,1)}};
      case YZ: return {{SignVec(0,1,0), SignVec(0,0,1)}};
      default: return {};
    }
  }

  // tet_face()
  DRAY_EXEC FaceTangents FaceTangents::tet_face(TetFaceTangent tangent_id)
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


  // loc()
  DRAY_EXEC Location FaceLocation::loc() const
  {
    return m_loc;
  }

  // tangents()
  DRAY_EXEC FaceTangents FaceLocation::tangents() const
  {
    return m_tangents;
  }

  // world_tangents()
  DRAY_EXEC void FaceLocation::world_tangents(
      const Vec<Vec<Float, 3>, 3> &jacobian,
      Vec<Float, 3> &t0,
      Vec<Float, 3> &t1) const
  {
    t0 = tangents().m_t[0].combine(jacobian);
    t1 = tangents().m_t[1].combine(jacobian);
  }

  // world_normal()
  DRAY_EXEC Vec<Float, 3> FaceLocation::world_normal(
      const Vec<Vec<Float, 3>, 3> &jacobian) const
  {
    Vec<Float, 3> t[2];
    world_tangents(jacobian, t[0], t[1]);
    return cross(t[0], t[1]).normalized();
  }



}//namespace dray

#endif//DRAY_FACE_LOCATION_HPP

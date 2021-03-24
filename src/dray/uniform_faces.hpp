// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_FACES_HPP
#define DRAY_UNIFORM_FACES_HPP

#include <dray/uniform_topology.hpp>

namespace dray
{
  struct UniformFaces
  {
    static UniformFaces from_uniform_topo(const UniformTopology &topo);

    int32 num_boundary_faces() const;
    int32 num_total_faces() const;

    void fill_total_faces(Vec<Float, 3> *face_centers_out) const;  // x-nrm, y-nrm, z-nrm

    enum FaceID : uint8 { X0 = 0, X1, Y0, Y1, Z0, Z1, NUM_FACES };
    int32 cell_idx_to_face_idx(int32 cell_idx, FaceID face_id) const;  // cell_idx: x varying fastest
    int32 cell_idx_to_face_idx(const Vec<int32, 3> &cell_xyz_idx, FaceID face_id) const;

    Vec<Float, 3> m_topo_spacing;
    Vec<Float, 3> m_topo_origin;
    Vec<int32, 3> m_topo_cell_dims;
  };
}

#endif//DRAY_UNIFORM_FACES_HPP

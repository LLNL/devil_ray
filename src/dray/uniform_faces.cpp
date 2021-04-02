// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/uniform_faces.hpp>

namespace dray
{
  //
  // from_uniform_topo()
  //
  UniformFaces UniformFaces::from_uniform_topo(const UniformTopology &topo)
  {
    UniformFaces uniform_faces;
    uniform_faces.m_topo_spacing = topo.spacing();
    uniform_faces.m_topo_origin = topo.origin();
    uniform_faces.m_topo_cell_dims = topo.cell_dims();
    return uniform_faces;
  }

  //
  // fill_total_faces(): assumes buffer ready, fill order is x-nrm, y-nrm, z-nrm
  //
  void UniformFaces::fill_total_faces(Vec<Float, 3> *face_centers_out) const
  {
    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];
    const Vec<Float, 3> &sp = m_topo_spacing;
    const Vec<Float, 3> &org = m_topo_origin;

    int32 index = 0;
    for (int32 kk = 0; kk < dims_z; ++kk)
      for (int32 jj = 0; jj < dims_y; ++jj)
        for (int32 plane_i = 0; plane_i < dims_x + 1; ++plane_i)
          face_centers_out[index++] = scale_and_offset(org, sp,
              {{(Float) plane_i, 0.5f + jj, 0.5f + kk}});

    for (int32 kk = 0; kk < dims_z; ++kk)
      for (int32 plane_j = 0; plane_j < dims_y + 1; ++plane_j)
        for (int32 ii = 0; ii < dims_x; ++ii)
          face_centers_out[index++] = scale_and_offset(org, sp,
              {{0.5f + ii, (Float) plane_j, 0.5f + kk}});

    for (int32 plane_k = 0; plane_k < dims_z + 1; ++plane_k)
      for (int32 jj = 0; jj < dims_y; ++jj)
        for (int32 ii = 0; ii < dims_x; ++ii)
          face_centers_out[index++] = scale_and_offset(org, sp,
              {{0.5f + ii, 0.5f + jj, (Float) plane_k}});
  }
}

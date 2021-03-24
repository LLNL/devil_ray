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
  // num_boundary_faces()
  //
  int32 UniformFaces::num_boundary_faces() const
  {
    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];

    return   2 * dims_y * dims_z    // +/- X-normal
           + 2 * dims_z * dims_x    // +/- Y-normal
           + 2 * dims_x * dims_y;   // +/- Z-normal
  }

  //
  // num_total_faces(): includes internal faces
  //
  int32 UniformFaces::num_total_faces() const
  {
    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];

    return   (dims_x + 1) * dims_y * dims_z   // X-normal
           + (dims_y + 1) * dims_z * dims_x   // Y-normal
           + (dims_z + 1) * dims_x * dims_y;  // Z-normal
  }


  static Vec<Float, 3> scale_and_offset(const Vec<Float, 3> &origin,
                                        const Vec<Float, 3> &scale,
                                        const Vec<Float, 3> &vec)
  {
    Vec<Float, 3> result = origin;
    for (int32 d = 0; d < 3; ++d)
      result[d] += vec[d] * scale[d];
    return result;
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

  //
  // cell_idx_to_face_idx()
  //
  int32 UniformFaces::cell_idx_to_face_idx(int32 cell_idx, FaceID face_id) const
  {
    /// fprintf(stdout, "\n\ncell_idx == %d\n", cell_idx);
    Vec<int32, 3> cell_xyz_idx = {{0, 0, 0}};
    for (int32 d = 0; d < 3; ++d)
    {
      cell_xyz_idx[d] = cell_idx % m_topo_cell_dims[d];
      cell_idx /= m_topo_cell_dims[d];
    }
    return this->cell_idx_to_face_idx(cell_xyz_idx, face_id);
  }

  //
  // cell_idx_to_face_idx()
  //
  int32 UniformFaces::cell_idx_to_face_idx(const Vec<int32, 3> &cell_xyz_idx, FaceID face_id) const
  {
    /// fprintf(stdout, "cell_xyz_idx == [%d, %d, %d]\n", cell_xyz_idx[0], cell_xyz_idx[1], cell_xyz_idx[2]);
    /// fprintf(stdout, "face_id == %d\n", int32(face_id));

    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];

    int32 plane_begin = 0;
    if (face_id == X0 || face_id == X1)
      plane_begin = 0;
    if (face_id == Y0 || face_id == Y1)
      plane_begin = (dims_x + 1) * dims_y * dims_z;
    if (face_id == Z0 || face_id == Z1)
      plane_begin = (dims_x + 1) * dims_y * dims_z + (dims_y + 1) * dims_z * dims_x;

    int32 normal_plane = 0;
    if (face_id == X0 || face_id == X1)
      normal_plane = 0;
    if (face_id == Y0 || face_id == Y1)
      normal_plane = 1;
    if (face_id == Z0 || face_id == Z1)
      normal_plane = 2;

    Vec<int32, 3> plane_dims_xyz = {{dims_x, dims_y, dims_z}};
    plane_dims_xyz[normal_plane] += 1;

    Vec<int32, 3> plane_offset_xyz = cell_xyz_idx;
    if (face_id == X1 || face_id == Y1 || face_id == Z1)
      plane_offset_xyz[normal_plane] += 1;

    int32 face_idx =   plane_begin
                     + plane_offset_xyz[0]
                     + plane_offset_xyz[1] * plane_dims_xyz[0]
                     + plane_offset_xyz[2] * plane_dims_xyz[0] * plane_dims_xyz[1];

    /// fprintf(stdout, "--> face_idx == %d\n", face_idx);

    return face_idx;
  }

}

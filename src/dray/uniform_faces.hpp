// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_FACES_HPP
#define DRAY_UNIFORM_FACES_HPP

#include <dray/uniform_topology.hpp>
#include <dray/exports.hpp>

namespace dray
{
  struct UniformFaces
  {
    enum FaceID : uint8 { X0 = 0, X1, Y0, Y1, Z0, Z1, NUM_FACES };
    static UniformFaces from_uniform_topo(const UniformTopology &topo);

    DRAY_EXEC static Vec<Float, 3> normal(FaceID face_id);
    DRAY_EXEC Float face_area(FaceID face_id) const;

    DRAY_EXEC int32 num_boundary_faces() const;
    DRAY_EXEC int32 num_total_faces() const;
    DRAY_EXEC int32 num_total_cells() const;

    void fill_total_faces(Vec<Float, 3> *face_centers_out) const;  // x-nrm, y-nrm, z-nrm

    DRAY_EXEC int32 cell_idx_to_face_idx(int32 cell_idx, FaceID face_id) const;  // cell_idx: x varying fastest
    DRAY_EXEC int32 cell_idx_to_face_idx(const Vec<int32, 3> &cell_xyz_idx, FaceID face_id) const;

    Vec<Float, 3> m_topo_spacing;
    Vec<Float, 3> m_topo_origin;
    Vec<int32, 3> m_topo_cell_dims;
  };
}


namespace dray
{
  //
  // normal()
  //
  DRAY_EXEC Vec<Float, 3> UniformFaces::normal(FaceID face_id)
  {
    Float unit = 1.0f;
    if (face_id == X0 || face_id == Y0 || face_id == Z0)
      unit = -1.0f;

    Vec<Float, 3> normal = {{0.0f, 0.0f, 0.0f}};
    if (face_id == X0 || face_id == X1)
      normal[0] = unit;
    else if (face_id == Y0 || face_id == Y1)
      normal[1] = unit;
    else
      normal[2] = unit;

    return normal;
  }

  //
  // face_area()
  //
  DRAY_EXEC Float UniformFaces::face_area(FaceID face_id) const
  {
    if (face_id == X0 || face_id == X1)
      return m_topo_spacing[1] * m_topo_spacing[2];
    if (face_id == Y0 || face_id == Y1)
      return m_topo_spacing[2] * m_topo_spacing[0];
    else
      return m_topo_spacing[0] * m_topo_spacing[1];
  }


  //
  // num_boundary_faces()
  //
  DRAY_EXEC int32 UniformFaces::num_boundary_faces() const
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
  DRAY_EXEC int32 UniformFaces::num_total_faces() const
  {
    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];

    return   (dims_x + 1) * dims_y * dims_z   // X-normal
           + (dims_y + 1) * dims_z * dims_x   // Y-normal
           + (dims_z + 1) * dims_x * dims_y;  // Z-normal
  }


  //
  // num_total_cells()
  //
  DRAY_EXEC int32 UniformFaces::num_total_cells() const
  {
    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];

    return dims_x * dims_y * dims_z;
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
  // cell_idx_to_face_idx()
  //
  DRAY_EXEC int32 UniformFaces::cell_idx_to_face_idx(int32 cell_idx, FaceID face_id) const
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
  DRAY_EXEC int32 UniformFaces::cell_idx_to_face_idx(const Vec<int32, 3> &cell_xyz_idx, FaceID face_id) const
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


#endif//DRAY_UNIFORM_FACES_HPP

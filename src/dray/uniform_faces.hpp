// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_FACES_HPP
#define DRAY_UNIFORM_FACES_HPP

#include <dray/uniform_topology.hpp>
#include <dray/exports.hpp>
#include <dray/face_location.hpp>

namespace dray
{
  struct QuadratureRule; //TODO replace with a more efficient quadrature rule

  struct UniformFaces
  {
    enum FaceID : uint8 { Z0 = 0, Z1, Y0, Y1, X0, X1, NUM_FACES };
    static UniformFaces from_uniform_topo(const UniformTopology &topo);

    DRAY_EXEC static Vec<Float, 3> normal(FaceID face_id);
    DRAY_EXEC static int32 nrml_axis(FaceID face_id);
    DRAY_EXEC Float face_area(FaceID face_id) const;

    DRAY_EXEC int32 num_boundary_faces() const;
    DRAY_EXEC int32 num_total_faces() const;
    DRAY_EXEC int32 num_total_cells() const;

    void fill_total_faces(Vec<Float, 3> *face_centers_out) const;  // z-nrm, y-nrm, x-nrm

    void fill_total_faces(FaceLocation *face_centers_out) const;

    void fill_total_faces(Vec<Float, 3> *face_points_out,
                          Float *face_weights_out,
                          const QuadratureRule &quadrature) const;

    DRAY_EXEC int32 cell_idx_to_face_idx(int32 cell_idx, FaceID face_id) const;  // cell_idx: x varying fastest
    DRAY_EXEC int32 cell_idx_to_face_idx(const Vec<int32, 3> &cell_xyz_idx, FaceID face_id) const;
    DRAY_EXEC int32 face_idx_to_nrml_axis(int32 face_idx) const;

    Vec<Float, 3> m_topo_spacing;
    Vec<Float, 3> m_topo_origin;
    Vec<int32, 3> m_topo_cell_dims;
  };

  struct UniformRelativeQuad
  {
    // interface
    static UniformRelativeQuad create(Float t0, Float t1, Float t2);
    Vec<Float, 3> project_point_to_quad(const Vec<Float, 3> &point,
                                        const Vec<Float, 3> &quad_center,
                                        int32 nrml_axis) const;
    void active(bool active);

    // implementation
    Float m_t[3];  // size in the normal direction should be 0
    bool m_active;
  };

  inline UniformRelativeQuad UniformRelativeQuad::create(
      Float t0, Float t1, Float t2)
  {
    UniformRelativeQuad ret{{t0, t1, t2}, true};
    return ret;
  }

  inline Vec<Float, 3> UniformRelativeQuad::project_point_to_quad(
      const Vec<Float, 3> &point,
      const Vec<Float, 3> &quad_center,
      int32 nrml_axis) const
  {
    if (m_active)
    {
      const Vec<Float, 3> diff = point - quad_center;
      Vec<Float, 3> y = quad_center;
      for (int32 i = 0; i < 3; ++i)
      {
        if (i == nrml_axis)
          continue;
        const Float T2 = m_t[i] * m_t[i];  // non-negative
        const Float dotted = diff[i] * m_t[i];  // dot(diff, tangent_i)
        const Float scale = (dotted > T2    ? 1.0f
                             : dotted < -T2 ? -1.0f
                             :                dotted/T2);
        y[i] += scale * m_t[i];  // y += scale * tangent_i
      }
      return y;
    }
    else
    {
      return quad_center;
    }
  }


  inline void UniformRelativeQuad::active(bool active)
  {
    m_active = active;
  }



  struct QuadratureRule  // subdivided midpoint rule (proxy)
  {
    // interface
    static const int32 MAX_DEGREE = 6;
    static const int32 s_points_per_degree[MAX_DEGREE + 1];

    static QuadratureRule create(int32 degree)
    {
      /// if (degree > MAX_DEGREE)
      ///   throw std::logic_error("degree exceeds MAX_DEGREE");
      return QuadratureRule{degree};
    }

    int32 points() const
    {
      /// return s_points_per_degree[m_degree];
      return m_degree + 1;
    }

    const Float * abscissas() const
    {
      /// return s_abscissas01 + s_offsets[m_degree];
      static std::vector<Float> absc;
      if (absc.size() != this->points())
      {
        absc.resize(this->points());
        for (int32 ii = 0; ii < this->points(); ++ii)
          absc[ii] = (ii + 0.5) / (m_degree + 1);
      }
      return absc.data();
    }

    const Float * weights() const
    {
      /// return s_weights + s_offsets[m_degree];
      static std::vector<Float> w;
      if (w.size() != this->points())
      {
        w.resize(this->points());
        for (int32 ii = 0; ii < this->points(); ++ii)
          w[ii] = 1.0 / (m_degree + 1);
      }
      return w.data();
    }

    // implementation
    int32 m_degree;

    static const int32 s_offsets[MAX_DEGREE + 1];
    static const Float s_abscissas01[];  // on the interval (0, 1)
    static const Float s_weights[];
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
  // nrml_axis()
  //
  DRAY_EXEC int32 UniformFaces::nrml_axis(FaceID face_id)
  {
    if (face_id == X0 || face_id == X1)
      return 0;
    else if (face_id == Y0 || face_id == Y1)
      return 1;
    else
      return 2;
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

    return   2 * dims_x * dims_y    // +/- Z-normal
           + 2 * dims_z * dims_x    // +/- Y-normal
           + 2 * dims_y * dims_z;   // +/- X-normal
  }

  //
  // num_total_faces(): includes internal faces
  //
  DRAY_EXEC int32 UniformFaces::num_total_faces() const
  {
    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];

    return + (dims_z + 1) * dims_x * dims_y   // Z-normal
           + (dims_y + 1) * dims_z * dims_x   // Y-normal
           + (dims_x + 1) * dims_y * dims_z;  // X-normal
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
    if (face_id == Z0 || face_id == Z1)
      plane_begin = 0;
    if (face_id == Y0 || face_id == Y1)
      plane_begin = (dims_z + 1) * dims_y * dims_x;
    if (face_id == X0 || face_id == X1)
      plane_begin = (dims_z + 1) * dims_y * dims_x + (dims_y + 1) * dims_z * dims_x;

    int32 normal_plane = 0;
    if (face_id == Z0 || face_id == Z1)
      normal_plane = 2;
    if (face_id == Y0 || face_id == Y1)
      normal_plane = 1;
    if (face_id == X0 || face_id == X1)
      normal_plane = 0;

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


  //
  // face_idx_to_nrml_axis()
  //
  DRAY_EXEC int32 UniformFaces::face_idx_to_nrml_axis(int32 face_idx) const
  {
    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];

    int32 plane_axis = 2;
    if (face_idx >= (dims_z + 1) * dims_y * dims_x)
      plane_axis = 1;
    if (face_idx >= (dims_z + 1) * dims_y * dims_x + (dims_y + 1) * dims_z * dims_x)
      plane_axis = 0;

    return plane_axis;
  }

}


#endif//DRAY_UNIFORM_FACES_HPP

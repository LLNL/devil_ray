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

    for (int32 plane_k = 0; plane_k < dims_z + 1; ++plane_k)
      for (int32 jj = 0; jj < dims_y; ++jj)
        for (int32 ii = 0; ii < dims_x; ++ii)
          face_centers_out[index++] = scale_and_offset(org, sp,
              {{0.5f + ii, 0.5f + jj, (Float) plane_k}});

    for (int32 kk = 0; kk < dims_z; ++kk)
      for (int32 plane_j = 0; plane_j < dims_y + 1; ++plane_j)
        for (int32 ii = 0; ii < dims_x; ++ii)
          face_centers_out[index++] = scale_and_offset(org, sp,
              {{0.5f + ii, (Float) plane_j, 0.5f + kk}});

    for (int32 kk = 0; kk < dims_z; ++kk)
      for (int32 jj = 0; jj < dims_y; ++jj)
        for (int32 plane_i = 0; plane_i < dims_x + 1; ++plane_i)
          face_centers_out[index++] = scale_and_offset(org, sp,
              {{(Float) plane_i, 0.5f + jj, 0.5f + kk}});
  }


  //
  // fill_total_faces(): with a quadrature set of a certain degree
  //                     instead of a single point per face,
  //                     use a surface quadrature per face.
  //
  void UniformFaces::fill_total_faces(
      Vec<Float, 3> *face_points_out,
      Float *face_weights_out,
      const QuadratureRule &quadrature) const
  {
    const int32 interval_points = quadrature.points();
    /// const int32 points_per_face = quadrature.points() * quadrature.points();
    const Float * q_abscissas = quadrature.abscissas();
    const Float * q_weights = quadrature.weights();

    const int32 dims_x = m_topo_cell_dims[0];
    const int32 dims_y = m_topo_cell_dims[1];
    const int32 dims_z = m_topo_cell_dims[2];
    const Vec<Float, 3> &sp = m_topo_spacing;
    const Vec<Float, 3> &org = m_topo_origin;

    int32 index = 0;
    for (int32 plane_k = 0; plane_k < dims_z + 1; ++plane_k)
      for (int32 jj = 0; jj < dims_y; ++jj)
        for (int32 ii = 0; ii < dims_x; ++ii)
          for (int32 quad_j = 0; quad_j < interval_points; ++quad_j)
            for (int32 quad_i = 0; quad_i < interval_points; ++quad_i)
            {
              face_points_out[index] = scale_and_offset(org, sp,
                  {{q_abscissas[quad_i] + ii, q_abscissas[quad_j] + jj, (Float) plane_k}});
              face_weights_out[index] = q_weights[quad_i] * q_weights[quad_j];
              index++;
            }

    for (int32 kk = 0; kk < dims_z; ++kk)
      for (int32 plane_j = 0; plane_j < dims_y + 1; ++plane_j)
        for (int32 ii = 0; ii < dims_x; ++ii)
          for (int32 quad_k = 0; quad_k < interval_points; ++quad_k)
            for (int32 quad_i = 0; quad_i < interval_points; ++quad_i)
            {
              face_points_out[index] = scale_and_offset(org, sp,
                  {{q_abscissas[quad_i] + ii, (Float) plane_j, q_abscissas[quad_k] + kk}});
              face_weights_out[index] = q_weights[quad_i] * q_weights[quad_k];
              index++;
            }

    for (int32 kk = 0; kk < dims_z; ++kk)
      for (int32 jj = 0; jj < dims_y; ++jj)
        for (int32 plane_i = 0; plane_i < dims_x + 1; ++plane_i)
          for (int32 quad_k = 0; quad_k < interval_points; ++quad_k)
            for (int32 quad_j = 0; quad_j < interval_points; ++quad_j)
            {
              face_points_out[index] = scale_and_offset(org, sp,
                  {{(Float) plane_i, q_abscissas[quad_j] + jj, q_abscissas[quad_k] + kk}});
              face_weights_out[index] = q_weights[quad_j] * q_weights[quad_k];
              index++;
            }
  }



  // QuadratureRule proxy implementation
  const int32 QuadratureRule::s_points_per_degree[MAX_DEGREE + 1]
    = { 1, 2, 3, 4, 5, 6, 7 };
  const int32 QuadratureRule::s_offsets[MAX_DEGREE + 1]
    = { 0, 1, 3, 6, 10, 16, 21 };
  const Float QuadratureRule::s_abscissas01[]  // on the interval (0, 1)
    = { (0+.5)/(0+1),
        (0+.5)/(1+1), (1+.5)/(1+1),
        (0+.5)/(2+1), (1+.5)/(2+1), (2+.5)/(2+1),
        (0+.5)/(3+1), (1+.5)/(3+1), (2+.5)/(3+1), (3+.5)/(3+1),
        (0+.5)/(4+1), (1+.5)/(4+1), (2+.5)/(4+1), (3+.5)/(4+1), (4+.5)/(4+1),
        (0+.5)/(5+1), (1+.5)/(5+1), (2+.5)/(5+1), (3+.5)/(5+1), (4+.5)/(5+1), (5+.5)/(5+1),
        (0+.5)/(6+1), (1+.5)/(6+1), (2+.5)/(6+1), (3+.5)/(6+1), (4+.5)/(6+1), (5+.5)/(6+1), (6+.5)/(6+1)
      };
  const Float QuadratureRule::s_weights[]
    = { 1./1,
        1./2, 1./2,
        1./3, 1./3, 1/.3,
        1./4, 1./4, 1./4, 1./4,
        1./5, 1./5, 1./5, 1./5, 1./5,
        1./6, 1./6, 1./6, 1./6, 1./6, 1./6,
        1./7, 1./7, 1./7, 1./7, 1./7, 1./7, 1./7
      };



}

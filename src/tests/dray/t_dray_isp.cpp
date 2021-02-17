// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/io/blueprint_uniform_topology.hpp>

#include <iostream>

#define EXAMPLE_MESH_SIDE_DIM 10


namespace dray
{
  Array<Vec<Float, 3>> boundary_face_centers(const UniformTopology &topo);

  // If the source is on the outside of the domain,
  // create rays from the source to the isps
  // and intersect the inflow boundary (closest to source).
  void intercept_inflow_boundary(const Array<Vec<Float, 3>> &isps,
                                 const UniformTopology &topo,
                                 const Vec<Float, 3> &src_pt,
                                 Array<Vec<int32, 3>> &inflow_cells,
                                 Array<Vec<Float, 3>> &inflow_pts);

  void point3d(const Array<Vec<Float, 3>> &points, std::ostream &out);
}



TEST (dray_hanus, dray_isp)
{

  /*
   * Mesh: Get inter-domain boundary faces
   *
   * Get global list of sources
   *  - list of source centers
   *
   * For each source:
   *   For each (outflow) boundary face:
   *     Create ray(source center --> face center)
   *       Trace ray backward from face center to an opposing domain boundary
   *       - for each intercepted cell, accumulate extinction
   *       // this is reducing segments that can be computed locally
   *
   *       Intersect ray/cone with inflow boundary faces, interpolation weights
   *       --> queue waiting on the inflow boundary face
   *
   *       Once read inflow boundary faces and interpolate,
   *       then populate outflow boundary faces and trigger MPI
   */

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            data);

  /// data["coordsets/coords/origin/z"] = double(0.0);

  /// std::cout << data.to_yaml() << "\n";

  conduit::Node verify_info;
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));


  
  std::shared_ptr<dray::UniformTopology> uni_topo
      = dray::detail::import_topology_into_uniform(data, data["coordsets/coords"]);

  std::ofstream outfile("boundary_points.3D");
  dray::Array<dray::Vec<dray::Float, 3>> bdry_points = boundary_face_centers(*uni_topo);
  dray::point3d(bdry_points, outfile);
  outfile.close();

  /// const conduit::Node &n_dims = coords["dims"];

}


namespace dray
{

  // from a local piece of the uniform domain,
  // get the centers of boundary faces.
  Array<Vec<Float, 3>> boundary_face_centers(const UniformTopology &topo)
  {
    Array<Vec<Float, 3>> face_centers;

    const Vec<int32,3> cell_dims = topo.cell_dims();
    const Vec<Float,3> origin = topo.origin();
    const Vec<Float,3> spacing = topo.spacing();

    size_t num_faces = 0;
    {
      num_faces += 2 * (cell_dims[1] * cell_dims[2]);  // normal = +/- X
      num_faces += 2 * (cell_dims[2] * cell_dims[0]);  // normal = +/- Y
      num_faces += 2 * (cell_dims[0] * cell_dims[1]);  // normal = +/- Z
    }

    face_centers.resize(num_faces);
    Vec<Float, 3> * face_centers_ptr = face_centers.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_faces), [=] DRAY_LAMBDA (int32 index)
    {
      Vec<int32, 2> face_dims;
      Vec<Float, 3> first_face_center;
      enum CubeFace {X_plus, X_minus, Y_plus, Y_minus, Z_plus, Z_minus};

      first_face_center[0] = origin[0] + spacing[0] * 0.5;
      first_face_center[1] = origin[1] + spacing[1] * 0.5;
      first_face_center[2] = origin[2] + spacing[2] * 0.5;

      CubeFace cube_face;
      int32 shifted_index = index;

      if (shifted_index < 2 * (cell_dims[1] * cell_dims[2]))
      {
        if (shifted_index < (cell_dims[1] * cell_dims[2]))
        {
          cube_face = X_plus;
          first_face_center[0] = origin[0] + cell_dims[0] * spacing[0];
        }
        else
        {
          cube_face = X_minus;
          first_face_center[0] = origin[0];
        }
        face_dims[0] = cell_dims[1];     // right hand rule looking down +X
        face_dims[1] = cell_dims[2];
      }
      else if ((shifted_index -= 2 * (cell_dims[1] * cell_dims[2])) < 2 * cell_dims[2] * cell_dims[0])
      {
        if (shifted_index < (cell_dims[2] * cell_dims[0]))
        {
          cube_face = Y_plus;
          first_face_center[1] = origin[1] + cell_dims[1] * spacing[1];
        }
        else
        {
          cube_face = Y_minus;
          first_face_center[1] = origin[1];
        }
        face_dims[0] = cell_dims[2];     // right hand rule looking down +Y
        face_dims[1] = cell_dims[0];
      }
      else if ((shifted_index -= 2 * (cell_dims[2] * cell_dims[0])) < 2 * cell_dims[0] * cell_dims[1])
      {
        if (shifted_index < (cell_dims[0] * cell_dims[1]))
        {
          cube_face = Z_plus;
          first_face_center[2] = origin[2] + cell_dims[2] * spacing[2];
        }
        else
        {
          cube_face = Z_minus;
          first_face_center[2] = origin[2];
        }
        face_dims[0] = cell_dims[0];     // right hand rule looking down +Z
        face_dims[1] = cell_dims[1];
      }

      int32 face_i = shifted_index % face_dims[0];
      int32 face_j = (shifted_index / face_dims[0]) % face_dims[1];

      // Apply right-hand rule to convert 2d index to 3d index.
      Vec<Float, 3> offset = {{0, 0, 0}};
      if (cube_face == X_plus || cube_face == X_minus)
      {
        offset[1] = face_i;
        offset[2] = face_j;
      }
      else if (cube_face == Y_plus || cube_face == Y_minus)
      {
        offset[2] = face_i;
        offset[0] = face_j;
      }
      else
      {
        offset[0] = face_i;
        offset[1] = face_j;
      }

      Vec<Float, 3> center = first_face_center;
      center[0] += spacing[0] * offset[0];
      center[1] += spacing[1] * offset[1];
      center[2] += spacing[2] * offset[2];

      face_centers_ptr[index] = center;
    });
    return face_centers;
  }

  void intercept_inflow_boundary(const Array<Vec<Float, 3>> &isps,
                                 const UniformTopology &topo,
                                 const Vec<Float, 3> &_src_pt,
                                 Array<Vec<int32, 3>> &inflow_cells,
                                 Array<Vec<Float, 3>> &inflow_pts)
  {
    const Vec<int32,3> cell_dims = topo.cell_dims();
    const Vec<Float,3> origin = topo.origin();
    const Vec<Float,3> spacing = topo.spacing();
    const Vec<Float, 3> src_pt = _src_pt;

    const size_t size = isps.size();
    inflow_cells.resize(size);
    inflow_pts.resize(size);
    const Vec<Float, 3> * isp_ptr = isps.get_device_ptr_const();
    Vec<int32, 3> * inflow_cell_ptr = inflow_cells.get_device_ptr();
    Vec<Float, 3> * inflow_pt_ptr = inflow_pts.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 index)
    {
      const Vec<Float, 3> isp = isp_ptr[index];
      Vec<Float, 3> dir_to_src = src_pt - isp;
      dir_to_src.normalize();

      // TODO voxel or face id?
      /// Vec<Float, 3> isp_relative = isp - origin;
      /// Vec<int32, 3> isp_voxel;
      /// isp_voxel[0] = isp_relative[0] / cell_dims[0];
      /// isp_voxel[1] = isp_relative[1] / cell_dims[1];
      /// isp_voxel[2] = isp_relative[2] / cell_dims[2];

      // Intersect the boundary between the isp and the source.
      // Assume the source is outside the domain.

      Vec<Float, 3> exit_planes = origin;
      if (dir_to_src[0] >= 0.0f) 
        exit_planes[0] += spacing[0] * cell_dims[0];
      if (dir_to_src[1] >= 0.0f) 
        exit_planes[1] += spacing[1] * cell_dims[1];
      if (dir_to_src[2] >= 0.0f) 
        exit_planes[2] += spacing[2] * cell_dims[2];

      Vec<Float, 3> exit_distances;
      exit_distances[0] = (exit_planes[0] - isp[0]) / dir_to_src[0];
      exit_distances[1] = (exit_planes[1] - isp[1]) / dir_to_src[1];
      exit_distances[2] = (exit_planes[2] - isp[2]) / dir_to_src[2];

      const Float hit_distance = min(min(exit_distances[0], exit_distances[1]), exit_distances[2]);
      const int hit_plane = (hit_distance == exit_distances[0] ? 0 :
                             hit_distance == exit_distances[1] ? 1 :
                                                                 2);
      Vec<Float, 3> exit = isp + dir_to_src * hit_distance;
      exit[hit_plane] = exit_planes[hit_plane];

      // TODO voxel or face id?

      //TODO return nearest neighbors
      inflow_pt_ptr[index] = exit;

    });
  }


  void point3d(const Array<Vec<Float, 3>> &points, std::ostream &out)
  {
    out << "X Y Z value" << std::endl;
    const Vec<Float, 3> * points_ptr = points.get_host_ptr_const();
    const size_t size = points.size();

    for (size_t ii = 0; ii < size; ++ii)
    {
      for (int d = 0; d < 3; ++d)
        out << points_ptr[ii][d] << " ";
      out << "0.";
      out << std::endl;
    }
  }

} // namespace dray




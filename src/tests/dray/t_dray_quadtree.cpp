// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "test_config.h"
#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/uniform_faces.hpp>
#include <dray/quadtree.cpp>

// uniform_cube()
std::shared_ptr<dray::UniformTopology>
uniform_cube(dray::Float spacing,
             dray::int32 cell_dims);

// uniform_stick()
std::shared_ptr<dray::UniformTopology>
uniform_stick(dray::Float spacing,
              dray::int32 cell_dims);

// uniform_face_centers()
dray::Array<dray::FaceLocation> uniform_face_centers(
    const dray::UniformTopology &mesh,
    dray::UniformFaces &face_map);


//
// dray_qt_unit_area
//
TEST(dray_quadtree, dray_qt_unit_area)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  std::shared_ptr<dray::UniformTopology> mesh = uniform_cube(1, 10);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_face_centers(*mesh, face_map);

  dray::QuadTreeForest forest;
  forest.resize(face_centers.size());

  dray::IntegrateToMesh integration_result =
      forest.integrate_phys_area_to_mesh(
        face_centers,
        mesh->jacobian_evaluator(),
        [=] DRAY_LAMBDA (const dray::FaceLocation &floc)  // integrand
  {
    return 1.0f;
  });

  EXPECT_FLOAT_EQ(face_map.num_total_faces(), integration_result.result());
}


//
// dray_qt_directional_area
//
TEST(dray_quadtree, dray_qt_directional_area)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  std::shared_ptr<dray::UniformTopology> mesh = uniform_cube(1, 10);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_face_centers(*mesh, face_map);

  dray::QuadTreeForest forest;
  forest.resize(face_centers.size());

  using dray::FaceTangents;

  dray::IntegrateToMesh integration_result =
      forest.integrate_phys_area_to_mesh(
        face_centers,
        mesh->jacobian_evaluator(),
        [=] DRAY_LAMBDA (const dray::FaceLocation &floc)  // integrand
  {
    if (floc.m_tangents == FaceTangents::cube_face_xy())
      return 1.0f;
    else if (floc.m_tangents == FaceTangents::cube_face_xz())
      return 3.0f;
    else
      return 5.0f;
  });

  EXPECT_FLOAT_EQ(face_map.num_total_faces() * (1.+3.+5.)/3., integration_result.result());
}


//
// dray_qt_stick_area
//
TEST(dray_quadtree, dray_qt_stick_area)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  // 2x.5x.5 cube with 4 segments in x
  std::shared_ptr<dray::UniformTopology> mesh = uniform_stick(.5, 4);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_face_centers(*mesh, face_map);

  dray::QuadTreeForest forest;
  forest.resize(face_centers.size());

  using dray::FaceTangents;

  dray::IntegrateToMesh integration_result =
      forest.integrate_phys_area_to_mesh(
        face_centers,
        mesh->jacobian_evaluator(),
        [=] DRAY_LAMBDA (const dray::FaceLocation &floc)  // integrand
  {
    if (floc.m_tangents == FaceTangents::cube_face_xy())
      return 1.0f;
    else if (floc.m_tangents == FaceTangents::cube_face_xz())
      return 3.0f;
    else
      return 5.0f;
  });

  EXPECT_FLOAT_EQ(
      5.0f*.25*(4+1) + 1.0f*1.*(1+1) + 3.0f*1.*(1+1),
      integration_result.result());
}


//TODO implement the adaptive quadtree,
//     add fill_boundary_faces(Array<FaceLocation>)
//     add face-normal geometry,
//     integrate some nonlinear flux function

/// //
/// // dray_qt_adaptive
/// //
/// TEST(dray_quadtree, dray_qt_adaptive)
/// {
///
///
/// }


//
// uniform_cube()
//
std::shared_ptr<dray::UniformTopology>
uniform_cube(dray::Float spacing,
             dray::int32 cell_dims)
{
  dray::Vec<dray::Float, 3> origin = {{0, 0, 0}};
  dray::Vec<dray::Float, 3> vspacing = {{spacing, spacing, spacing}};
  dray::Vec<dray::int32, 3> vcell_dims = {{cell_dims, cell_dims, cell_dims}};

  return std::make_shared<dray::UniformTopology>
      (vspacing, origin, vcell_dims);
}

//
// uniform_stick()
//
std::shared_ptr<dray::UniformTopology>
uniform_stick(dray::Float spacing,
              dray::int32 cell_dims)
{
  dray::Vec<dray::Float, 3> origin = {{0, 0, 0}};
  dray::Vec<dray::Float, 3> vspacing = {{spacing, spacing, spacing}};
  dray::Vec<dray::int32, 3> vcell_dims = {{cell_dims, 1, 1}};

  return std::make_shared<dray::UniformTopology>
      (vspacing, origin, vcell_dims);
}


//
// uniform_face_centers()
//
dray::Array<dray::FaceLocation> uniform_face_centers(
    const dray::UniformTopology &mesh,
    dray::UniformFaces &face_map)
{
  dray::Array<dray::FaceLocation> face_centers;

  face_map = dray::UniformFaces::from_uniform_topo( mesh);
  face_centers.resize(face_map.num_total_faces());
  face_map.fill_total_faces(face_centers.get_host_ptr());

  return face_centers;
}

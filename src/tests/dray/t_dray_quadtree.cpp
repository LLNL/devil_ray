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
#include <dray/pagani.hpp>

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

// uniform_boundary_face_centers()
dray::Array<dray::FaceLocation> uniform_boundary_face_centers(
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


//
// dray_qt_unit_surface_area
//
TEST(dray_quadtree, dray_qt_unit_surface_area)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  // 2x.5x.5 cube with 4 segments in x
  std::shared_ptr<dray::UniformTopology> mesh = uniform_stick(.5, 4);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_boundary_face_centers(*mesh, face_map);

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

  EXPECT_FLOAT_EQ(.25*2 + 1.*2 + 1.*2, integration_result.result());
}


//
// dray_qt_static_boundary_flux
//
TEST(dray_quadtree, dray_qt_static_boundary_flux)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;
  using dray::pi;

  const int32 N = 256;
  std::shared_ptr<dray::UniformTopology> mesh = uniform_cube(1./N, N);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_boundary_face_centers(*mesh, face_map);

  dray::QuadTreeForest forest;
  forest.resize(face_centers.size());

  const Float source = 13;  // arbitrary point source term

  const dray::UniformTopology::Evaluator xyz = mesh->evaluator();
  const dray::UniformTopology::JacobianEvaluator jacobian = mesh->jacobian_evaluator();

  dray::IntegrateToMesh integration_result =
      forest.integrate_phys_area_to_mesh(
        face_centers,
        jacobian,
        [=] DRAY_LAMBDA (const dray::FaceLocation &floc)  // integrand
  {
    Vec<Float, 3> face_normal = floc.world_normal(jacobian(floc.loc()));

    const Vec<Float, 3> r = xyz(floc.loc()) - Vec<Float, 3>{{.5,.5,.5}};
    const Float mag2 = r.magnitude2();

    if (dot(face_normal, r) < 0)
      face_normal = -face_normal;

    const Vec<Float, 3> field = r.normalized() * source / mag2;

    const Float grand = dot(field, face_normal);
    return grand;
  });

  // The result is close to correct with N between 256 and 1024.
  // With larger N the result successively shrinks.
  // E.g. with N=2^14 the result is 4, way below the correct value of 163.36.
  // Probably this is due to precision issues and summation order.
  EXPECT_NEAR(4 * pi() * source, integration_result.result(), 0.05);
}


//
// dray_qt_uniform_refn_boundary_flux
//
TEST(dray_quadtree, dray_qt_uniform_refn_boundary_flux)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;
  using dray::pi;

  const int32 N = 1;
  const int32 M = 256;
  std::shared_ptr<dray::UniformTopology> mesh = uniform_cube(1./N, N);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_boundary_face_centers(*mesh, face_map);

  // Initialize quadtree forest on mesh boundary faces.
  dray::QuadTreeForest forest;
  forest.resize(face_centers.size());

  // Refine quadtree forest until each face has MxM quadrants.
  dray::Array<int32> refinements;
  for (int32 m = 1; m < M; m *= 2)
  {
    refinements.resize(forest.num_nodes());
    array_memset(refinements, 1);
    forest.execute_refinements(refinements);
  }

  const Float source = 13;  // arbitrary point source term

  const dray::UniformTopology::Evaluator xyz = mesh->evaluator();
  const dray::UniformTopology::JacobianEvaluator jacobian = mesh->jacobian_evaluator();

  dray::IntegrateToMesh integration_result =
      forest.integrate_phys_area_to_mesh(
        face_centers,
        jacobian,
        [=] DRAY_LAMBDA (const dray::FaceLocation &floc)  // integrand
  {
    Vec<Float, 3> face_normal = floc.world_normal(jacobian(floc.loc()));

    const Vec<Float, 3> r = xyz(floc.loc()) - Vec<Float, 3>{{.5,.5,.5}};
    const Float mag2 = r.magnitude2();

    if (dot(face_normal, r) < 0)
      face_normal = -face_normal;

    const Vec<Float, 3> field = r.normalized() * source / mag2;

    const Float grand = dot(field, face_normal);
    return grand;
  });

  EXPECT_NEAR(4 * pi() * source, integration_result.result(), 0.05);
}

//TODO implement the adaptive quadtree,
//     integrate some nonlinear flux function

//
// dray_qt_adaptive
//
TEST(dray_quadtree, dray_qt_adaptive)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "qt_adaptive");

  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const Float tol_rel = 1e-3;
  const int32 iter_max = 25;
  const int32 nodes_max = 1e+8;

  const int32 N = 1;
  std::shared_ptr<dray::UniformTopology> mesh = uniform_cube(1./N, N);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_boundary_face_centers(*mesh, face_map);

  // Initialize quadtree forest on mesh boundary faces.
  dray::QuadTreeForest forest;
  forest.resize(face_centers.size());

  const Float source = 13;  // arbitrary point source term

  const dray::UniformTopology::Evaluator xyz = mesh->evaluator();
  const dray::UniformTopology::JacobianEvaluator jacobian = mesh->jacobian_evaluator();
  const auto integrand =
      [=] DRAY_LAMBDA (const dray::FaceLocation &floc)  // integrand
      {
        Vec<Float, 3> face_normal = floc.world_normal(jacobian(floc.loc()));

        const Vec<Float, 3> r = xyz(floc.loc()) - Vec<Float, 3>{{.5,.5,.5}};
        const Float mag2 = r.magnitude2();

        if (dot(face_normal, r) < 0)
          face_normal = -face_normal;

        const Vec<Float, 3> field = r.normalized() * source / mag2;

        const Float grand = dot(field, face_normal);
        return grand;
      };

  // PaganiIteration<Jacobian, Integrand>
  auto pagani = dray::pagani_iteration(
      face_centers,
      jacobian,
      integrand,
      tol_rel,
      nodes_max,
      iter_max
  );

  const bool output_quadtree = true;

  remove_test_file(output_file);

  conduit::Node bp_dataset;
  const std::string extension = ".blueprint_root";  // visit sees time series if use json format.
  char cycle_suffix[8] = "_000000";

  // export and save mesh
  if (output_quadtree)
  {
    pagani.forest().to_blueprint(
        face_centers,
        xyz,
        integrand,
        bp_dataset);
    conduit::relay::io::blueprint::save_mesh(bp_dataset, output_file + std::string(cycle_suffix) + extension);
  }

  int32 level = 0;
  while (pagani.need_more_refinements())
  {
    pagani.execute_refinements();

    level++;
    if (output_quadtree)
    {
      snprintf(cycle_suffix, 8, "_%06d", level);
      // export and save mesh
      pagani.forest().to_blueprint(
          face_centers,
          xyz,
          integrand,
          bp_dataset);
      conduit::relay::io::blueprint::save_mesh(bp_dataset, output_file + std::string(cycle_suffix) + extension);
    }
  }

  dray::IntegrateToMesh integration_result;
  integration_result.m_result = pagani.value_error().value();

  EXPECT_NEAR(4 * dray::pi() * source, integration_result.result(), 0.05);
}


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

//
// uniform_boundary_face_centers()
//
dray::Array<dray::FaceLocation> uniform_boundary_face_centers(
    const dray::UniformTopology &mesh,
    dray::UniformFaces &face_map)
{
  dray::Array<dray::FaceLocation> face_centers;

  face_map = dray::UniformFaces::from_uniform_topo( mesh);
  face_centers.resize(face_map.num_boundary_faces());
  face_map.fill_boundary_faces(face_centers.get_host_ptr());

  return face_centers;
}

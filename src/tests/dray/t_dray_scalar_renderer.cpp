// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/filters/mesh_boundary.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/rendering/scalar_renderer.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/rendering/surface.hpp>

#include <dray/import_order_policy.hpp>

#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

void setup_camera (dray::Camera &camera)
{
  camera.set_width (512);
  camera.set_height (512);

  dray::Vec<dray::float32, 3> pos;
  pos[0] = .5f;
  pos[1] = -1.5f;
  pos[2] = .5f;
  camera.set_up (dray::make_vec3f (0, 0, 1));
  camera.set_pos (pos);
  camera.set_look_at (dray::make_vec3f (0.5, 0.5, 0.5));
}

TEST (dray_scalar_renderer, dray_scalars)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "slice_scalars");
  remove_test_image (output_file);

  std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_001860.root";

  dray::DataSet dataset = dray::BlueprintReader::load (root_file, dray::ImportOrderPolicy::general());

  dray::Camera camera;
  setup_camera (camera);

  dray::Vec<float, 3> point;
  point[0] = 0.5f;
  point[1] = 0.5f;
  point[2] = 0.5f;

  std::cout<<dataset.field_info();
  // dray::Vec<float,3> normal;
  std::shared_ptr<dray::SlicePlane> slicer
    = std::make_shared<dray::SlicePlane>(dataset);
  slicer->field("velocity_y");
  slicer->point(point);
  dray::ColorMap color_map("thermal");
  slicer->color_map(color_map);

  dray::ScalarRenderer renderer;
  renderer.set(slicer);
  renderer.field_names(dataset.fields());
  dray::ScalarBuffer sb = renderer.render(camera);

  conduit::Node mesh;
  sb.to_node(mesh);
  conduit::relay::io_blueprint::save(mesh, output_file + ".blueprint_root_hdf5");
}

TEST (dray_scalar_renderer, dray_triple_surface)
{
  std::string root_file = std::string(DATA_DIR) + "tripple_point/field_dump.cycle_006700.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "triple_scalar");
  remove_test_image (output_file);

  dray::DataSet dataset = dray::BlueprintReader::load (root_file, dray::ImportOrderPolicy::general());

  dray::MeshBoundary boundary;
  dray::DataSet faces = boundary.execute(dataset);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth (-60);
  camera.reset_to_bounds (dataset.topology()->bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("density");

  dray::ScalarRenderer renderer;
  renderer.set(surface);
  renderer.field_names(dataset.fields());
  dray::ScalarBuffer sb = renderer.render(camera);

  conduit::Node mesh;
  sb.to_node(mesh);
  conduit::relay::io_blueprint::save(mesh, output_file + ".blueprint_root_hdf5");
}

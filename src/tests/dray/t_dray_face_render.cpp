// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_faces, dray_taylor_green)
{
  std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_001860.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file = conduit::utils::join_file_path (output_path, "tg_faces");
  remove_test_image (output_file);

  dray::DataSet dataset = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::DataSet faces = boundary.execute(dataset);

  dray::ColorTable color_table ("Spectral");

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.topology()->bounds());
  camera.azimuth(-30.f);
  camera.elevate(35.f);

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("density");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);

  fb.save(output_file);
  //EXPECT_TRUE (check_test_image (output_file));
  //fb.save_depth (output_file + "_depth");
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}
#if 0
TEST (dray_faces, dray_impeller_faces)
{
  std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "impeller_faces");
  remove_test_image (output_file);

  dray::DataSet dataset = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::DataSet faces = boundary.execute(dataset);

  dray::ColorTable color_table ("Spectral");

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.topology()->bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("diffusion");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file));
  fb.save_depth (output_file + "_depth");
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}


TEST (dray_faces, dray_warbly_faces)
{
  std::string root_file = std::string (DATA_DIR) + "warbly_cube/warbly_cube_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "warbly_faces");
  remove_test_image (output_file);

  dray::DataSet dataset = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::DataSet faces = boundary.execute(dataset);

  dray::ColorTable color_table ("Spectral");

  color_table.clear ();
  dray::Vec<float, 3> white{ 1.f, 1.f, 1.f };
  color_table.add_point (0.f, white);
  color_table.add_point (1.f, white);

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  dray::Vec<float32, 3> center = { 0.500501, 0.510185, 0.495425 };
  dray::Vec<float32, 3> v_normal = { -0.706176, 0.324936, -0.629072 };
  dray::Vec<float32, 3> v_pos = center + v_normal * -1.92;


  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);


  dray::Vec<float32, 3> pos = { -.30501f, 1.50185f, 2.37722f };
  dray::Vec<float32, 3> look_at = { -0.5f, 1.0, -0.5f };
  camera.set_look_at (look_at);
  dray::Vec<float32, 3> up = { 0.f, 0.f, 1.f };
  camera.set_up (up);
  camera.reset_to_bounds (dataset.topology()->bounds());


  dray::Vec<float32, 3> top = { 0.500501, 1.510185, 0.495425 };
  dray::Vec<float32, 3> mov = top - camera.get_pos ();
  mov.normalize ();

  dray::PointLight light;
  light.m_pos = pos + mov * 3.f;
  light.m_amb = { 0.5f, 0.5f, 0.5f };
  light.m_diff = { 0.70f, 0.70f, 0.70f };
  light.m_spec = { 0.9f, 0.9f, 0.9f };
  light.m_spec_pow = 90.0;

  dray::Vec<float,4> line_color = { 0.0f, 0.0f, 0.0f, 0.5f };
  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("bananas");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_color(line_color);

  dray::Renderer renderer;
  renderer.add(surface);
  renderer.add_light(light);
  dray::Framebuffer fb = renderer.render(camera);

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file));
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}
#endif

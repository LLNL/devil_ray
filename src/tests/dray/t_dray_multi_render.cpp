// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/camera.hpp>
#include <dray/filters/slice.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/shaders.hpp>
#include <dray/ray_tracing/renderer.hpp>
#include <dray/ray_tracing/slice_plane.hpp>
#include <dray/ray_tracing/contour.hpp>

void setup_camera (dray::Camera &camera)
{
  camera.set_width (1024);
  camera.set_height (1024);

  dray::Vec<dray::float32, 3> pos;
  pos[0] = .5f;
  pos[1] = -1.5f;
  pos[2] = .5f;
  camera.set_up (dray::make_vec3f (0, 0, 1));
  camera.set_pos (pos);
  camera.set_look_at (dray::make_vec3f (0.5, 0.5, 0.5));
}

TEST (dray_multi_render, dray_simple)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "multi_render");
  remove_test_image (output_file);

  //std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_001860.root";
  std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_000190.root";

  dray::DataSet dataset = dray::BlueprintReader::load (root_file);

  dray::Camera camera;
  camera.reset_to_bounds(dataset.topology()->bounds());
  camera.azimuth(-40);
  camera.elevate(-40);
  //setup_camera (camera);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::PointLight plight;
  plight.m_pos = { 1.2f, -0.15f, 0.4f };
  plight.m_amb = { 0.3f, 0.3f, 0.3f };
  plight.m_diff = { 0.70f, 0.70f, 0.70f };
  plight.m_spec = { 0.30f, 0.30f, 0.30f };
  //plight.m_pos = { 1.2f, -0.15f, 0.4f };
  //plight.m_amb = { 1.0f, 1.0f, 1.f };
  //plight.m_diff = { 0.0f, 0.0f, 0.0f };
  //plight.m_spec = { 0.0f, 0.0f, 0.0f };
  plight.m_spec_pow = 90.0;

  dray::AABB<3> bounds = dataset.topology()->bounds();

  dray::Vec<float, 3> point;
  point[0] = bounds.center()[0];
  point[1] = bounds.center()[1];
  point[2] = bounds.center()[2];

  std::cout<<dataset.field_info();
  // dray::Vec<float,3> normal;
  std::shared_ptr<dray::ray_tracing::SlicePlane> slicer
    = std::make_shared<dray::ray_tracing::SlicePlane>(dataset);
  //slicer->field("specific_internal_energy");
  slicer->field("velocity_y");
  slicer->point(point);
  dray::ColorMap color_map("thermal");
  slicer->color_map(color_map);

  std::shared_ptr<dray::ray_tracing::Contour> contour
    = std::make_shared<dray::ray_tracing::Contour>(dataset);
  contour->field("density");
  contour->iso_field("velocity_y");
  contour->iso_value(0.09);
  contour->color_map(color_map);

  dray::ray_tracing::Renderer renderer;
  renderer.add(slicer);
  renderer.add(contour);
  renderer.add_light(plight);
  dray::Framebuffer fb = renderer.render(camera);

  //dray::Slice slicer;
  //slicer.set_field ("velocity_y");
  //slicer.set_point (point);

  //slicer.execute (rays, dataset, framebuffer);

  //framebuffer.save (output_file);
  fb.save (output_file);
  fb.save_depth("depth");
  EXPECT_TRUE (check_test_image (output_file));
}

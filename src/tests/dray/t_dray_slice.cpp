// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"
#include <dray/io/blueprint_reader.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/utils/appstats.hpp>

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

TEST (dray_slice, dray_slice)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "slice");
  remove_test_image (output_file);

  std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_001860.root";

  dray::DataSet dataset = dray::BlueprintReader::load (root_file);

  dray::Camera camera;
  setup_camera (camera);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::PointLight plight;
  plight.m_pos = { 1.2f, -0.15f, 0.4f };
  plight.m_amb = { 1.0f, 1.0f, 1.f };
  plight.m_diff = { 0.0f, 0.0f, 0.0f };
  plight.m_spec = { 0.0f, 0.0f, 0.0f };
  plight.m_spec_pow = 90.0;

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

  dray::Renderer renderer;
  renderer.add(slicer);
  renderer.add_light(plight);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  dray::stats::StatStore::write_point_stats ("locate_stats");
  EXPECT_TRUE (check_test_image (output_file));
}

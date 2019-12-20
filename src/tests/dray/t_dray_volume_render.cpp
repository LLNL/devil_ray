// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/filters/volume_integrator.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/mfem_reader.hpp>

#include <dray/camera.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_volume_render, dray_volume_render_simple)
{
  std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "impeller_vr");
  remove_test_image (output_file);

  dray::DataSet dataset = dray::BlueprintReader::nload (root_file);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.00f);
  color_table.add_alpha (0.2f, 0.01f);
  color_table.add_alpha (0.3f, 0.09f);
  color_table.add_alpha (0.4f, 0.01f);
  color_table.add_alpha (0.5f, 0.01f);
  color_table.add_alpha (0.6f, 0.41f);
  color_table.add_alpha (0.7f, 0.49f);
  color_table.add_alpha (0.8f, 0.41f);
  color_table.add_alpha (0.9f, 0.41f);
  color_table.add_alpha (1.0f, 0.4f);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;
  dray::Framebuffer framebuffer (c_width, c_height);
  framebuffer.clear ();

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);

  camera.reset_to_bounds (dataset.topology()->bounds());
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::VolumeIntegrator integrator;
  integrator.set_field ("diffusion");
  integrator.set_color_table (color_table);
  integrator.execute (rays, dataset, framebuffer);

  framebuffer.save (output_file);
  EXPECT_TRUE (check_test_image (output_file));
}

TEST (dray_volume_render, dray_volume_render_triple)
{
  std::string root_file = std::string(DATA_DIR) + "tripple_point/field_dump.cycle_006700.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "triple_vr");
  remove_test_image (output_file);

  dray::DataSet dataset = dray::BlueprintReader::nload (root_file);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.00f);
  color_table.add_alpha (0.2f, 0.01f);
  color_table.add_alpha (0.3f, 0.09f);
  color_table.add_alpha (0.4f, 0.21f);
  color_table.add_alpha (0.5f, 0.31f);
  color_table.add_alpha (0.6f, 0.41f);
  color_table.add_alpha (0.7f, 0.59f);
  color_table.add_alpha (0.8f, 0.69f);
  color_table.add_alpha (0.9f, 0.61f);
  color_table.add_alpha (1.0f, 0.6f);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth (-60);
  camera.reset_to_bounds (dataset.topology()->bounds());

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::Framebuffer framebuffer (c_width, c_height);
  framebuffer.clear ();

  dray::VolumeIntegrator integrator;
  integrator.set_field ("density");
  integrator.set_color_table (color_table);
  integrator.execute (rays, dataset, framebuffer);

  framebuffer.save (output_file);
  EXPECT_TRUE (check_test_image (output_file));
}

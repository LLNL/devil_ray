// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/filters/to_bernstein.hpp>
#include <dray/filters/mesh_boundary.hpp>

#include <dray/rendering/camera.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/synthetic/spiral_sample.hpp>


TEST (dray_to_bernstein_filter, dray_to_bernstein_filter)
{
  dray::DataSet dataset_raw = dray::SynthesizeSpiralSample(1, 0.5, 2, 20).synthesize();

  dray::DataSet dataset = dray::ToBernstein().execute(dataset_raw);

  using DummyFieldHex = dray::Field<dray::Element<3, 1, dray::Tensor, -1>>;
  dataset.add_field(std::make_shared<DummyFieldHex>( DummyFieldHex::uniform_field(
          dataset.topology()->cells(), dray::Vec<float,1>{{0}}, "uniform")));

  dray::MeshBoundary boundary;
  dray::DataSet faces = boundary.execute(dataset);

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "diy_spiral");
  remove_test_image (output_file);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.elevate(0);
  camera.azimuth(0);

  camera.reset_to_bounds (faces.topology()->bounds());

  dray::ColorTable color_table ("ColdAndHot");

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("uniform");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.05);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
}

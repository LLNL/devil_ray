// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/filters/isosurfacing.hpp>

#include <dray/rendering/camera.hpp>
#include <dray/rendering/contour.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/synthetic/affine_radial.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/error.hpp>

TEST (dray_isosurface_filter, dray_isosurface_filter)
{
  const dray::Vec<int, 3> extents = {{4, 4, 4}};
  const dray::Vec<float, 3> origin = {{0.0f, 0.0f, 0.0f}};
  const dray::Vec<float, 3> radius = {{1.0f, 1.0f, 1.0f}};
  const dray::Vec<float, 3> range_radius = {{1.0f, 1.0f, 1.0f}};

  dray::DataSet dataset =
      dray::SynthesizeAffineRadial(extents, origin, radius)
      .equip("perfection", range_radius)
      .synthesize();

  const float isoval = 0.9;

  std::shared_ptr<dray::ExtractIsosurface> iso_extractor
    = std::make_shared<dray::ExtractIsosurface>();
  iso_extractor->iso_field("perfection");
  iso_extractor->iso_value(isoval);

  /// dray::DataSet isosurf = iso_extractor->execute(dataset);

  dray::DataSet isoblocks = iso_extractor->execute(dataset);
  std::cout << "input dataset contains " << dataset.topology()->cells() << " cells.\n";
  std::cout << "isoblocks dataset contains " << isoblocks.topology()->cells() << " cells.\n";

  { using namespace dray;
    DRAY_INFO("Extracted as surface. Now to render.");
  }

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "isoblock_spherical");
  remove_test_image (output_file);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(-40);

  camera.reset_to_bounds (dataset.topology()->bounds());

  dray::ColorTable color_table ("ColdAndHot");
  // dray::Vec<float,3> normal;

  /// std::shared_ptr<dray::Contour> contour
  ///   = std::make_shared<dray::Contour>(dataset);
  std::shared_ptr<dray::Contour> contour
    = std::make_shared<dray::Contour>(isoblocks);
  contour->field("perfection");
  contour->iso_field("perfection");
  contour->iso_value(isoval);
  contour->color_map().color_table(color_table);;

  dray::Renderer renderer;
  renderer.add(contour);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);

  { using namespace dray;
    DRAY_ERROR("TODO render it");
  }
}

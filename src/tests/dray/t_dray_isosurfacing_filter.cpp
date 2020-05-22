// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/filters/isosurfacing.hpp>
#include <dray/filters/to_bernstein.hpp>

#include <dray/rendering/camera.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/synthetic/affine_radial.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/error.hpp>

TEST (dray_isosurface_filter, dray_isosurface_filter)
{
  const dray::Vec<int, 3> extents = {{4, 4, 4}};
  const dray::Vec<float, 3> origin = {{0.0f, 0.0f, 0.0f}};
  const dray::Vec<float, 3> radius = {{1.0f, 1.0f, 1.0f}};
  const dray::Vec<float, 3> range_radius = {{1.0f, 1.0f, -1.0f}};

  dray::Collection collxn =
      dray::SynthesizeAffineRadial(extents, origin, radius)
      .equip("perfection", range_radius)
      .synthesize();

  const float isoval = 1.1;

  std::shared_ptr<dray::ExtractIsosurface> iso_extractor
    = std::make_shared<dray::ExtractIsosurface>();
  iso_extractor->iso_field("perfection");
  iso_extractor->iso_value(isoval);

  // Extract isosurface. Partly made of tris, partly quads.
  auto isosurf_tri_quad = iso_extractor->execute(collxn);
  dray::Collection isosurf_tris = isosurf_tri_quad.first;
  dray::Collection isosurf_quads = isosurf_tri_quad.second;

  isosurf_quads = dray::ToBernstein().execute(isosurf_quads);
  //TODO convert tris

  size_t count_cells = 0;
  for (dray::DataSet &ds : collxn.domains())
    count_cells += ds.topology()->cells();
  std::cout << "input collxn contains " << count_cells << " cells.\n";

  count_cells = 0;
  for (dray::DataSet &ds : isosurf_tris.domains())
    count_cells += ds.topology()->cells();
  std::cout << "isosurf_tris collxn contains " << count_cells << " cells.\n";

  count_cells = 0;
  for (dray::DataSet &ds : isosurf_quads.domains())
    count_cells += ds.topology()->cells();
  std::cout << "isosurf_quads collxn contains " << count_cells << " cells.\n";

  // Add a field so that it can be rendered.
  using DummyFieldTri = dray::Field<dray::Element<2, 1, dray::Simplex, -1>>;
  using DummyFieldQuad = dray::Field<dray::Element<2, 1, dray::Tensor, -1>>;
  for (dray::DataSet &ds : isosurf_tris.domains())
    ds.add_field(std::make_shared<DummyFieldTri>( DummyFieldTri::uniform_field(
            ds.topology()->cells(), dray::Vec<float,1>{{0}}, "uniform")));
  for (dray::DataSet &ds : isosurf_quads.domains())
    ds.add_field(std::make_shared<DummyFieldQuad>( DummyFieldQuad::uniform_field(
            ds.topology()->cells(), dray::Vec<float,1>{{0}}, "uniform")));

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "isosurface_meshed");
  remove_test_image (output_file);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(-40);

  camera.reset_to_bounds (collxn.bounds());

  dray::ColorTable color_table ("ColdAndHot");

  std::shared_ptr<dray::Surface> surface_tris
    = std::make_shared<dray::Surface>(isosurf_tris);
  std::shared_ptr<dray::Surface> surface_quads
    = std::make_shared<dray::Surface>(isosurf_quads);
  surface_tris->field("uniform");
  surface_tris->color_map().color_table(color_table);
  surface_tris->draw_mesh (false);
  surface_tris->line_thickness(.1);
  surface_quads->field("uniform");
  surface_quads->color_map().color_table(color_table);
  surface_quads->draw_mesh (false);
  surface_quads->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface_tris);
  renderer.add(surface_quads);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
}

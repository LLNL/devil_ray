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
#include <dray/rendering/world_annotator.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/line_renderer.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/math.hpp>

#include <dray/dray.hpp>

#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace dray;

#if 0
TEST (dray_faces, dray_crop_lines_no_crop)
{
  Vec<int32, 2> p1, p2;
  int32 width, height;

  width = height = 3;
  p1[0] = 0;
  p1[1] = 0;
  p2[0] = 2;
  p2[1] = 2;

  // we expect the line to be cropped to (0,0) -> (2,2)
  crop_line_to_bounds(p1, p2, width, height);

  EXPECT_EQ(p1[0], 0);
  EXPECT_EQ(p1[1], 0);
  EXPECT_EQ(p2[0], 2);
  EXPECT_EQ(p2[1], 2);
}

TEST (dray_faces, dray_crop_lines)
{
  Vec<int32, 2> p1, p2;
  int32 width, height;

  width = height = 3;
  p1[0] = 0;
  p1[1] = 0;
  p2[0] = 4;
  p2[1] = 2;

  // we expect the line to be cropped to (0,0) -> (2,2)
  crop_line_to_bounds(p1, p2, width, height);

  EXPECT_EQ(p1[0], 0);
  EXPECT_EQ(p1[1], 0);
  EXPECT_EQ(p2[0], 2);
  EXPECT_EQ(p2[1], 1);
}

TEST (dray_faces, dray_crop_lines_corners)
{
  Vec<int32, 2> p1, p2;
  int32 width, height;

  width = height = 3;
  p1[0] = 0;
  p1[1] = 0;
  p2[0] = 3;
  p2[1] = 3;

  // we expect the line to be cropped to (0,0) -> (2,2)
  crop_line_to_bounds(p1, p2, width, height);

  EXPECT_EQ(p1[0], 0);
  EXPECT_EQ(p1[1], 0);
  EXPECT_EQ(p2[0], 2);
  EXPECT_EQ(p2[1], 2);
}

#endif
TEST (dray_faces, dray_world_annotator)
{
  std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file = "world_stuff";
  // conduit::utils::join_file_path (output_path, "lines_test");
  remove_test_image (output_file);

  Collection dataset = BlueprintReader::load (root_file);

  MeshBoundary boundary;
  Collection faces = boundary.execute(dataset);

  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  camera.azimuth(-25);
  camera.elevate(7);
  camera.set_zoom(1.5);
  //camera.set_up(((Vec<float32, 3>) {{0.1f, 1.f, 0.1f}}).normalized());
  //camera.set_pos(camera.get_pos() - 10.f * camera.get_look_at());

  ColorTable color_table ("Spectral");

  Framebuffer fb;

  std::shared_ptr<Surface> surface
      = std::make_shared<Surface>(faces);
  surface->field("diffusion");
  surface->color_map().color_table(color_table);
  Renderer renderer;
  renderer.add(surface);
  //renderer.triad(true);
  fb = renderer.render(camera);

  AABB<3> aabb = dataset.bounds();
  WorldAnnotator wannot(aabb);
  wannot.render(fb, camera);

  fb.composite_background();
  fb.save(output_file);
  fb.save_depth(output_file + "_depth");
}

//TEST (dray_faces, dray_world_annotator)
//{
//  std::string output_file = "line_zoom";
//  remove_test_image (output_file);
//
//  // Camera
//  const int c_width  = 1024;
//  const int c_height = 1024;
//
//  AABB<3> bounds;
//  Vec<float32,3> p1 = {{0.f, 0.f, 0.f}};
//  Vec<float32,3> p2 = {{1.f, 1.f, 0.f}};
//  bounds.include(p1);
//  bounds.include(p2);
//
//  Camera camera;
//  camera.set_width (c_width);
//  camera.set_height (c_height);
//  camera.reset_to_bounds (bounds);
//  camera.set_zoom(1.5);
//
//  Framebuffer fb;
//
//
//  fb.composite_background();
//  fb.save(output_file);
//  fb.save_depth(output_file + "_depth");
//}


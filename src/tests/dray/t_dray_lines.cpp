// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <dray/rendering/renderer.hpp>
#include <dray/rendering/line_renderer.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace dray;

void generate_lines(
  Array<Vec<float32,3>> &starts, 
  Array<Vec<float32,3>> &ends, 
  int num_lines,
  const int width,
  const int height)
{
  starts.resize(num_lines);
  ends.resize(num_lines);

  Vec<float32,3> *starts_ptr = starts.get_host_ptr();
  Vec<float32,3> *ends_ptr = ends.get_host_ptr();

  srand(time(NULL));

  for (int i = 0; i < num_lines; i ++)
  {

    int x1 = rand() % width;
    int y1 = rand() % height;
    int x2 = rand() % width;
    int y2 = rand() % height;

    starts_ptr[i] = {{(float) x1, (float) y1, 0.f}};
    ends_ptr[i] = {{(float) x2, (float) y2, 0.f}};
  }
}

TEST (dray_faces, dray_impeller_faces)
{
  std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file = "hereiam";
  // conduit::utils::join_file_path (output_path, "lines_test");
  // remove_test_image (output_file);

  Collection dataset = BlueprintReader::load (root_file);

  ColorTable color_table ("Spectral");

  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  AABB<3> aabb = dataset.bounds();

  for (int i = 0; i < 5; i ++)
  {
    int num_lines = 1000;
    Array<Vec<float32,3>> starts;
    Array<Vec<float32,3>> ends;
    Matrix<float32, 3, 3> transform;
    transform.identity();
    generate_lines(starts, ends, num_lines, c_width, c_height);

    dray::Framebuffer fb1;
    dray::Framebuffer fb2;
    LineRenderer lines;

    Vec<float32,3> *starts_ptr = starts.get_host_ptr();
    Vec<float32,3> *ends_ptr = ends.get_host_ptr();

    lines.render(fb1, transform, starts, ends);
    lines.justinrender(fb2, transform, starts, ends);

    std::cout << "==========" << std::endl;
    
    fb1.save(output_file + "1");
    fb2.save(output_file + "2");
    // fb.save_depth (output_file + "_depth");
  }
}

TEST (dray_faces, dray_aabb)
{
  std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file = "hereiam";
  // conduit::utils::join_file_path (output_path, "lines_test");
  // remove_test_image (output_file);

  Collection dataset = BlueprintReader::load (root_file);

  ColorTable color_table ("Spectral");

  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  AABB<3> aabb = dataset.bounds();

  std::cout << aabb << std::endl;

  
}


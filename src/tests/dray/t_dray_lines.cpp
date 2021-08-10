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

using namespace dray;
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
  int num_lines = 3;
  Array<Vec<float32,3>> starts;
  Array<Vec<float32,3>> ends;
  starts.resize(num_lines);
  ends.resize(num_lines);

  Vec<float32,3> *starts_ptr = starts.get_host_ptr();
  Vec<float32,3> *ends_ptr = ends.get_host_ptr();

  starts_ptr[0] = {{400.1232f, 500.546f,0.557f}};
  starts_ptr[1] = {{33.345f, 900.34f, 0.435f}};
  starts_ptr[2] = {{500.43434f, 0.787f, 0.213f}};
  ends_ptr[0] = {{800.543f, 1100.775f,0.375f}};
  ends_ptr[1] = {{900.357f, 33.7835f, 0.1235f}};
  ends_ptr[2] = {{500.564f, 1000.75543f, 0.778f}};

  dray::Framebuffer fb;
  LineRenderer lines;

  lines.render(fb, starts, ends);

  fb.save(output_file);
  fb.save_depth (output_file + "_depth");
}


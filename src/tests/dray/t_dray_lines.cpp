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
  std::string output_file =
  conduit::utils::join_file_path (output_path, "lines_test");
  remove_test_image (output_file);

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
  int num_lines = 1;
  Array<Vec<float32,3>> starts;
  Array<Vec<float32,3>> ends;
  starts.resize(num_lines);
  ends.resize(num_lines);

  Vec<float32,3> *starts_ptr = starts.get_host_ptr();
  Vec<float32,3> *ends_ptr = ends.get_host_ptr();

  dray::Framebuffer fb;
  LineRenderer lines;

  fb.save(output_file);
  fb.save_depth (output_file + "_depth");
}


// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <dray/queries/intersect.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_faces, dray_warbly_faces)
{
  std::string root_file = std::string (DATA_DIR) + "warbly_cube/warbly_cube_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "warbly_faces");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  std::vector<dray::Vec<double,3>> ips;
  ips.resize(1);
  ips[0][0] = .33;
  ips[0][1] = .33;
  ips[0][2] = .33;

  std::vector<dray::Vec<double,3>> dirs;
  dirs.resize(1);
  dirs[0][0] = .5;
  dirs[0][1] = .5;
  dirs[0][2] = .5;

  dray::Array<int> face_ids;
  dray::Array<dray::Vec<dray::Float,3>> res_ips;
  dray::Intersect intersector;
  intersector.execute(dataset, dirs, ips, face_ids, res_ips);


  //dray::stats::StatStore::write_ray_stats (c_width, c_height);
}

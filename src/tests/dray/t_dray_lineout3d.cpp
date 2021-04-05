// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>
#include <dray/queries/lineout_3d.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

using namespace dray;
TEST (dray_scalar_renderer, dray_scalars)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "slice_scalars");
  remove_test_image (output_file);

  std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_001860.root";

  Collection collection = dray::BlueprintReader::load (root_file);

  Lineout3D lineout;

  lineout.samples(10);
  lineout.add_var("density");
  // the data set bounds are [0,1] on each axis
  Vec<Float,3> start = {{0.01f,0.5f,0.5f}};
  Vec<Float,3> end = {{0.99f,0.5f,0.5f}};
  lineout.add_line(start, end);

  Lineout3D::Result res = lineout.execute(collection);
  for(int i = 0; i < res.m_values[0].size(); ++i)
  {
    std::cout<<"Value "<<i<<" "<<res.m_values[0].get_value(i)<<"\n";
  }

}

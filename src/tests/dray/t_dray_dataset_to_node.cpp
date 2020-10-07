// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>
#include <dray/dray_node_to_dataset.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_reflect, dray_reflect_2d)
{
  std::string root_file = std::string (DATA_DIR) + "taylor_green_2d.cycle_000050.root";
  std::string output_path = prepare_output_dir ();

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::DataSet domain = collection.domain(0);;
  conduit::Node n_dataset;
  domain.to_node(n_dataset);

  dray::to_dataset(n_dataset);

}

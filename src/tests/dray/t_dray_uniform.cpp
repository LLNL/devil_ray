// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <conduit_blueprint.hpp>

#include <dray/math.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/queries/lineout.hpp>

#include <fstream>
#include <stdlib.h>

using namespace dray;

int EXAMPLE_MESH_SIDE_DIM = 20;

TEST (dray_uniform, dray_uniform_lineout)
{
  std::string output_path = prepare_output_dir ();

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);


  dray::DataSet domain = dray::BlueprintLowOrder::import(data);
  dray::Collection collection;
  collection.add_domain(domain);

  Lineout lineout;

  lineout.samples(10);
  lineout.add_var("braid");
  // the data set bounds are [0,1] on each axis
  Vec<Float,3> start = {{0.01f,0.5f,0.5f}};
  Vec<Float,3> end = {{0.99f,0.5f,0.5f}};
  lineout.add_line(start, end);

  Lineout::Result res = lineout.execute(collection);
  for(int i = 0; i < res.m_values[0].size(); ++i)
  {
    std::cout<<"Value "<<i<<" "<<res.m_values[0].get_value(i)<<"\n";
  }

}


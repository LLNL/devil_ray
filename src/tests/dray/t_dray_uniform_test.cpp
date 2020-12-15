// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"
#include <dray/io/blueprint_reader.hpp>
#include <dray/filters/path_lengths.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#define EXAMPLE_MESH_SIDE_DIM 10

void set_up(conduit::Node &data)
{
  const int size = data["fields/radial/values"].dtype().number_of_elements();
  double *radial = data["fields/radial/values"].value();
  double minv = 1e30, maxv = -1e30;
  for(int i = 0; i < size; ++i)
  {
    double val = radial[i];
    minv = std::min(minv, val);
    maxv = std::max(maxv, val);
  }

  data["fields/emission"] = data["fields/radial"];
  data["fields/absorption"] = data["fields/radial"];

  double *emission= data["fields/emission/values"].value();
  double *absorption= data["fields/absorption/values"].value();

  for(int i = 0; i < size; ++i)
  {
    double val = radial[i];
    val = (val - minv) / (maxv - minv);
    absorption[i] = val * 0.5f;
    emission[i] = val;
  }

}

TEST (dray_bananas, dray_cool_beans)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "uniform");
  remove_test_image (output_file);

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            data);
  set_up(data);
  //data.print();
  dray::DataSet dataset = dray::BlueprintReader::blueprint_to_dray_uniform(data);
  //conduit::relay::io::save(data,"uniform", "hdf5");

  dray::Vec<float,3> detector_center;
  detector_center[0] = 0.f;
  detector_center[1] = 0.f;
  detector_center[2] = -15.f;

  dray::PathLengths pl;
  pl.resolution(100,100);
  pl.absorption_field("absorption");
  pl.emission_field("emission");
  pl.point(detector_center);
  pl.execute(dataset);

}

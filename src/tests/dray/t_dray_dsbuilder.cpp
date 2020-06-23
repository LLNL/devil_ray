// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/utils/dataset_builder.hpp>
#include <conduit.hpp>
#include <conduit_relay.hpp>

TEST (dray_dsbuilder, dray_dsbuilder_simple)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "dsbuilder_simple_bp");

  std::vector<std::string> vfield_names = {"red", "blue"};
  std::vector<std::string> efield_names = {"green"};

  dray::DataSetBuilder dsbuilder(dray::DataSetBuilder::Hex, vfield_names, efield_names);
  dray::HexRecord hex_record = dsbuilder.new_empty_hex_record();

  const dray::HexVData<dray::Float, 3> test_coords = {{ {{0,0,0}},
                                                        {{1,0,0}},
                                                        {{0,1,0}},
                                                        {{1,1,0}},
                                                        {{0,0,1}},
                                                        {{1,0,1}},
                                                        {{0,1,1}},
                                                        {{1,1,1}} }};
  hex_record.coords(test_coords);
  hex_record.scalar_vdata("red",  {{ {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}} }});
  hex_record.scalar_vdata("blue", {{ {{0}}, {{1}}, {{0}}, {{1}}, {{10}}, {{12}}, {{14}}, {{16}} }});
  hex_record.scalar_edata("green", {{ {{101}} }});

  dsbuilder.add_hex_record(hex_record);

  conduit::Node mesh;
  dsbuilder.to_blueprint(mesh, "coords");
  conduit::relay::io_blueprint::save(mesh, output_file + ".blueprint_root_hdf5");
}

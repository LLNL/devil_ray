// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <dray/io/blueprint_reader.hpp>
#include <dray/filters/first_scatter.hpp>

#define EXAMPLE_MESH_SIDE_DIM 10

// No longer needed once get kripke data
void set_up(conduit::Node &data, int num_moments)
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

  conduit::DataType emission_dtype = data["fields/radial/values"].dtype();
  emission_dtype.set_number_of_elements(num_moments * size);

  data["fields/absorption"] = data["fields/radial"];
  data["fields/emission"] = data["fields/radial"];
  data["fields/emission/values"] = conduit::Node(emission_dtype);

  double *emission= data["fields/emission/values"].value();
  double *absorption= data["fields/absorption/values"].value();

  for(int i = 0; i < size; ++i)
  {
    double val = radial[i];
    val = (val - minv) / (maxv - minv);
    absorption[i] = val * 0.5f;
    emission[num_moments * i] = val;

    // Add zeros in between the isotropic emission values.
    for (int j = 1; j < num_moments; ++j)
      emission[num_moments * i + j] = 0.0f;
  }
}

TEST (dray_first_scatter, dray_absorption)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "first_scatter");
  remove_test_image (output_file);

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            EXAMPLE_MESH_SIDE_DIM,
                                            data);

  const int legendre_order = 3;
  const int num_moments = (legendre_order+1) * (legendre_order+1);

  set_up(data, num_moments);
  //data.print();
  dray::DataSet dataset = dray::BlueprintReader::blueprint_to_dray_uniform(data);
  //conduit::relay::io::save(data,"uniform", "hdf5");

  dray::FirstScatter integrator;
  integrator.legendre_order(legendre_order);
  integrator.total_cross_section_field("absorption");
  integrator.emission_field("emission");
  integrator.overwrite_first_scatter_field("emission");
  integrator.execute(dataset);
}

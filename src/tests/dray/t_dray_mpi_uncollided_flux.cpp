// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/dray.hpp>
#include <dray/filters/matt_scatter.hpp>

/// #include <dray/io/blueprint_reader.hpp>
#include <dray/io/blueprint_moments.hpp>
#include <dray/io/array_mapping.hpp>

#include <conduit_blueprint.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>

#include <mpi.h>


TEST(aton_dray, aton_import_and_integrate)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (DATA_DIR) + "kripke_data.root";

  conduit::Node data;
  conduit::relay::mpi::io::blueprint::load_mesh(root_file, data, comm);

  dray::int32 num_moments;
  dray::detail::ArrayMapping amap;
  dray::Collection dray_collection = dray::detail::import_into_uniform_moments(data, amap, num_moments);


  dray::UncollidedFlux first_scatter;
  first_scatter.emission_field("phi");
  first_scatter.total_cross_section_field("sigt");
  first_scatter.legendre_order(sqrt(num_moments) - 1);
  first_scatter.overwrite_first_scatter_field("phi_uc");

  first_scatter.uniform_isotropic_scattering(0.05f);  // TODO don't assume uniform scattering

  first_scatter.execute(dray_collection);
}

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}

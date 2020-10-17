// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/dray.hpp>
#include <dray/filters/volume_balance.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <mpi.h>

TEST (dray_redistribute, redistribute)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (DATA_DIR) + "laghos_tg.cycle_000350.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
    conduit::utils::join_file_path (output_path, "balance");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  int rank = dray::dray::mpi_rank();
  int size = dray::dray::mpi_size();

  dray::VolumeBalance balancer;
  dray::Collection res = balancer.execute(dataset);

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

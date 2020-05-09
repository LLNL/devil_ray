// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/dray.hpp>
#include <mpi.h>

TEST (dray_mpi_smoke, dray_about)
{
  //
  // Set Up MPI
  //
  int par_rank;
  int par_size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &par_rank);
  MPI_Comm_size(comm, &par_size);

  std::cout<<"Rank "
              << par_rank
              << " of "
              << par_size
              << " reporting\n";

  if(par_rank == 0)
  {
    dray::dray tracer;
    tracer.about ();
  }

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

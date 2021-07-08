//
// Copyright (c) 2014-19, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#include <Kripke/SteadyStateSolver.h>
#include <Kripke.h>
#include <Kripke/Core/Comm.h>
#include <Kripke/ConduitInterface.h>
#include <Kripke/Kernel.h>
#include <Kripke/ParallelComm.h>
#include <Kripke/Core/PartitionSpace.h>
#include <Kripke/Timing.h>
#include <Kripke/SweepSolver.h>
#include <vector>
#include <stdio.h>

#include <Kripke/ConduitInterface.h>

using namespace Kripke::Core;

// TODO move these includes to aton
#include <dray/dray.hpp>
#include <dray/filters/first_scatter.hpp>
#include <dray/io/blueprint_moments.hpp>
#include <dray/io/array_mapping.hpp>

namespace aton
{
  /**
   * Raytraces source in the field first_scatter_name, computes scattering,
   * and overwrites the field with the result.
   */
  void raytrace(Kripke::Core::DataStore &data_store,
                const std::string &sigt_name,
                const std::string &source_name,
                const std::string &uncollided_flux_name);
}

/**
  Run solver iterations.
*/
int Kripke::SteadyStateSolver (Kripke::Core::DataStore &data_store, size_t max_iter, bool use_first_scatter, bool block_jacobi)
{
  KRIPKE_TIMER(data_store, Solve);

  PartitionSpace &pspace = data_store.getVariable<PartitionSpace>("pspace");

  Kripke::Core::Comm const &comm = data_store.getVariable<Kripke::Core::Comm>("comm");
  if(comm.rank() == 0){
    printf("\n");
    printf("Steady State Solve\n");
    printf("==================\n\n");
  }

  // Intialize unknowns
  Kripke::Kernel::kConst(data_store.getVariable<Kripke::Field_Flux>("psi"), 0.0);

  // aton::ComputeSourceScatter
  if (use_first_scatter)
  {
    Kripke::Kernel::kConst(data_store.getVariable<Kripke::Field_Moments>("phi"), 0.0);
    Kripke::Kernel::source(data_store, "phi");
#ifdef KRIPKE_USE_MPI
    dray::dray::mpi_comm(comm.comm());
#endif
    aton::raytrace(data_store, "sigt", "phi", "phi_uc");  // Uncollided flux moments
  }

  // Counting particles
  if (use_first_scatter)
  {
    Kripke::Kernel::kConst(data_store.getVariable<Kripke::Field_Flux>("rhs"), 0.0);
    Kripke::Kernel::LPlusTimes(data_store, "rhs", "phi_uc");  // temp rhs: the other flux variable
    Kripke::Kernel::kAdd(data_store.getVariable<Kripke::Field_Flux>("psi"),
                         data_store.getVariable<Kripke::Field_Flux>("rhs"));

    double part = Kripke::Kernel::population(data_store);
    if(comm.rank() == 0){
      printf("  Uncollided: particle count=%e\n", part);
      fflush(stdout);
    }
  }


  // Loop over iterations
  double part_last = 0.0;
  for(size_t iter = 0;iter < max_iter;++ iter){


    /*
     * Compute the RHS:  rhs = LPlus*S*L*psi + Q
     */


    // Discrete to Moments transformation (phi = L*psi)
    Kripke::Kernel::kConst(data_store.getVariable<Field_Moments>("phi"), 0.0);
    Kripke::Kernel::LTimes(data_store);


    // Scattering and source, in moment space.
    if (use_first_scatter)
    {
      // Add uncollided flux moments to phi.
      Kripke::Kernel::kAdd(data_store.getVariable<Kripke::Field_Moments>("phi"),
                           data_store.getVariable<Kripke::Field_Moments>("phi_uc"));

      // Compute Scattering Source Term (phi_out = S*(phi + phi_UC))
      // Note: S*phi_UC == first_scatter source
      Kripke::Kernel::kConst(data_store.getVariable<Kripke::Field_Moments>("phi_out"), 0.0);
      Kripke::Kernel::scattering(data_store);
    }
    else
    {
      // Compute Scattering Source Term (psi_out = S*phi)
      Kripke::Kernel::kConst(data_store.getVariable<Kripke::Field_Moments>("phi_out"), 0.0);
      Kripke::Kernel::scattering(data_store);

      // Add original source to phi after scattering operator.
      Kripke::Kernel::source(data_store, "phi_out");
    }


    // Moments to Discrete transformation (rhs = LPlus*phi_out)
    Kripke::Kernel::kConst(data_store.getVariable<Kripke::Field_Flux>("rhs"), 0.0);
    Kripke::Kernel::LPlusTimes(data_store);




    /*
     * Sweep (psi = Hinv*rhs)
     */
    {
      // Create a list of all groups
      int num_subdomains = pspace.getNumSubdomains(SPACE_PQR);
      std::vector<SdomId> sdom_list(num_subdomains);
      for(SdomId i{0};i < num_subdomains;++ i){
        sdom_list[*i] = i;
      }

      // Sweep everything
      Kripke::SweepSolver(data_store, sdom_list, block_jacobi);
    }



    /*
     * Population edit and convergence test
     */
    double part = Kripke::Kernel::population(data_store);
    if(comm.rank() == 0){
      printf("  iter %d: particle count=%e, change=%e\n", (int)iter, part, (part-part_last)/part);
      fflush(stdout);
    }
    part_last = part;



  }

  if (use_first_scatter)
  {
    // Then psi solves H[psi] == (LPlus S L)[psi] + first_scatter
    // This is the collided flux.
    // Need to add the uncollided flux, psi = psi_C + psi_UC

    // Compute psi_UC = LTimes * phi_UC
    Kripke::Kernel::kConst(data_store.getVariable<Kripke::Field_Flux>("rhs"), 0.0);
    Kripke::Kernel::LPlusTimes(data_store, "rhs", "phi_uc");  // temp rhs: the other flux variable

    // Compute psi = psi_C = psi_UC
    Kripke::Kernel::kAdd(data_store.getVariable<Kripke::Field_Flux>("psi"),
                         data_store.getVariable<Kripke::Field_Flux>("rhs"));

  }

  fprintf(stderr, "Finished!\n");

  double part = Kripke::Kernel::population(data_store);
  if(comm.rank() == 0){
    printf("  Final: particle count=%e\n", part);
    fflush(stdout);
  }

  // VisDump exports phi, so compute phi from the latest psi.
  Kripke::Kernel::kConst(data_store.getVariable<Field_Moments>("phi"), 0.0);
  Kripke::Kernel::LTimes(data_store);

  // write out the solution to a vis dump
  VisDump(data_store);

  if(comm.rank() == 0){
    printf("  Solver terminated\n");
  }

  return(0);
}


namespace aton
{
  //
  // raytrace()
  //
  void raytrace(Kripke::Core::DataStore &data_store,
                const std::string &sigt_name,
                const std::string &source_name,
                const std::string &uf_name)
  {
    conduit::Node conduit_dataset;
    Kripke::ToBlueprint(data_store, conduit_dataset);

    dray::int32 num_moments;
    dray::detail::ArrayMapping amap;
    dray::Collection dray_collection = dray::detail::import_into_uniform_moments(conduit_dataset, amap, num_moments);

    dray::FirstScatter first_scatter;
    first_scatter.emission_field(source_name);
    first_scatter.total_cross_section_field(sigt_name);
    first_scatter.legendre_order(sqrt(num_moments) - 1);
    first_scatter.uniform_isotropic_scattering(1.0f);  // TODO don't assume uniform scattering
    first_scatter.return_type(dray::FirstScatter::ReturnUncollidedFlux);

    first_scatter.overwrite_first_scatter_field(uf_name);
    first_scatter.execute(dray_collection);

    dray::detail::export_from_uniform_moments(dray_collection, amap, conduit_dataset);

    // Because ToBlueprint does zero-copy, the results are in the data_store now.
  }
}


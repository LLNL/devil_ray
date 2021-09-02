// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/uniform_topology.hpp>
#include <dray/data_model/low_order_field.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <iostream>
#include <random>


dray::Collection egg_cartons(
    const std::string &field_name,
    dray::float64 sigmat_amplitude,
    dray::float64 sigmat_period,
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain);

//
// main()
//
int main (int argc, char *argv[])
{
  init_furnace();

  std::string config_file = "";
  std::string output_file = "egg_carton";

  if (argc != 2)
  {
    std::cout << "Missing configure file name\n";
    exit (1);
  }

  config_file = argv[1];

  Config config (config_file);
  /// config.load_data ();
  /// config.load_camera ();
  /// config.load_field ();
  /// dray::Collection collection = config.m_collection;

  double sigmat_amplitude = 1;
  double sigmat_period = 1;
  if (config.m_config.has_child("sigmat_amplitude"))
    sigmat_amplitude = config.m_config["sigmat_amplitude"].to_double();
  if (config.m_config.has_child("sigmat_period"))
    sigmat_period = config.m_config["sigmat_period"].to_double();
  printf("sigmat_amplitude:%f\nsigmat_period:%f\n",
      sigmat_amplitude, sigmat_period);

  using dray::Vec;
  using dray::Float;
  using dray::int32;
  Vec<Float, 3> global_origin = {{0, 0, 0}};
  Vec<Float, 3> spacing = {{1./64, 1./64, 1./64}};
  Vec<int32, 3> domains = {{4, 4, 4}};
  Vec<int32, 3> cell_dims = {{16, 16, 16}};

  dray::Collection collection = egg_cartons(
      "sigt",
      sigmat_amplitude,
      sigmat_period,
      global_origin,
      spacing,
      domains,
      cell_dims);

  // Output to blueprint for visit.
  conduit::Node conduit_collection;
  for (int dom = 0; dom < collection.local_size(); ++dom)
  {
    conduit::Node & conduit_domain = conduit_collection.append();
    dray::DataSet domain = collection.domain(dom);
    domain.to_blueprint(conduit_domain);
    conduit_domain["state/domain_id"] = domain.domain_id();
  }
  conduit::relay::io::blueprint::save_mesh(
      conduit_collection, output_file + ".blueprint_root_hdf5");

  //TODO 1.   Specify locations of interpolation surfaces
  //TODO 2.   Modify/duplicate the first_scatter filter to accept multidomains
  //            and interpolation surfaces
  //TODO 3.   Compute the leakage
  //TODO 4.   Compute the positivity

  finalize_furnace();
}



dray::Collection egg_cartons(
    const std::string &field_name,
    dray::float64 sigmat_amplitude,
    dray::float64 sigmat_period,
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain)
{
  using namespace dray;

  Collection collection;

  // Mesh.
  int32 domain_id = 0;
  int32 domain_i[3];
  int32 global_cell[3];
  for (domain_i[2] = 0; domain_i[2] < domains_layout[2]; ++domain_i[2])
  {
    global_cell[2] = domain_i[2] * cells_per_domain[2];
    for (domain_i[1] = 0; domain_i[1] < domains_layout[1]; ++domain_i[1])
    {
      global_cell[1] = domain_i[1] * cells_per_domain[1];
      for (domain_i[0] = 0; domain_i[0] < domains_layout[0]; ++domain_i[0])
      {
        global_cell[0] = domain_i[0] * cells_per_domain[0];

        // domain_origin
        Vec<Float, 3> domain_origin = origin;
        for (int d = 0; d < 3; ++d)
          domain_origin[d] += spacing[d] * global_cell[d];

        // Initialize mesh.
        // domain spacing and cell dims are "spacing" and "cells_per_domain"
        std::shared_ptr<UniformTopology> mesh =
            std::make_shared<UniformTopology>(
                spacing, domain_origin, cells_per_domain);
        mesh->name("topo");

        // Add domain to collection.
        DataSet domain(mesh);
        domain.domain_id(domain_id);
        collection.add_domain(domain);

        domain_id++;
      }
    }
  }

  // Add egg carton field.
  const auto sigmat_cell_avg = [=] (
      const Vec<Float, 3> &lo,
      const Vec<Float, 3> &hi,
      float64 amplitude,
      float64 period)
  {
    float64 product = 1;
    for (int d = 0; d < 3; ++d)
    {
      // Definite integral avg of sin^2(pi * x[d] / p) / (width[d])
      float64 factor =
          (0.5
          + period / (4 * pi() * (hi[d] - lo[d]))
            * (sin(lo[d] * 2 * pi() / period) - sin(hi[d] * 2 * pi() / period)));
      product *= factor;
    }
    return amplitude * product;
  };

  const auto add_field_to_domain = [=, &sigmat_cell_avg](DataSet &domain)
  {
    UniformTopology * mesh = dynamic_cast<UniformTopology *>(domain.mesh()); 
    const Vec<int32, 3> dims = mesh->cell_dims();
    const Vec<Float, 3> spacing = mesh->spacing();
    const Vec<Float, 3> origin = mesh->origin();

    int32 size = dims[0] * dims[1] * dims[2];

    Array<Float> sigt;
    sigt.resize(size);
    Float * sigt_ptr = sigt.get_host_ptr();

    int32 i[3];
    int32 global_cell[3];
    for (i[2] = 0; i[2] < dims[2]; ++i[2])
      for (i[1] = 0; i[1] < dims[1]; ++i[1])
        for (i[0] = 0; i[0] < dims[0]; ++i[0])
        {
          const int32 offset = i[0] + dims[0] * (i[1] + dims[1] * i[2]);
          Vec<Float, 3> xmin = {{i[0] * spacing[0],
                                 i[1] * spacing[1],
                                 i[2] * spacing[2]}};
          xmin += origin;
          const Vec<Float, 3> xmax = xmin + spacing;

          Float value = sigmat_cell_avg(
              xmin, xmax, sigmat_amplitude, sigmat_period);
          sigt_ptr[offset] = value;
        }

    std::shared_ptr<LowOrderField> field =
        std::make_shared<LowOrderField>(
            sigt, LowOrderField::Assoc::Element, dims);
    field->name(field_name);
    domain.add_field(field);
  };

  for (DataSet &domain : collection.domains())
    add_field_to_domain(domain);

  return collection;
}






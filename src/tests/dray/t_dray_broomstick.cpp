// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/io/blueprint_uniform_topology.hpp>
#include <dray/GridFunction/low_order_field.hpp>
#include <dray/filters/first_scatter.hpp>

#include <iostream>


class Broomstick
{
  public:
    void set_up(conduit::Node &data);
    void check_ucflux_volume_src(dray::DataSet &dataset);
    void check_ucflux_point_src(dray::DataSet &dataset);

    // properties
    void cells_length(size_t cells_length) { m_cells_length = cells_length; }
    void length_x(double length_x) { m_length_x = length_x; }
    void absorption(const std::string &absorption) { m_absorption = absorption; }
    void emission(const std::string &emission)     { m_emission = emission; }
    void ucflux(const std::string &ucflux)         { m_ucflux = ucflux; }
    void neutrons_per_second(double neutrons_per_second) { m_neutrons_per_second = neutrons_per_second; }
    void absorption_per_cm(double absorption_per_cm)    { m_absorption_per_cm = absorption_per_cm; }

    double spacing_dx() const { return this->length_x() / this->cells_length(); }
    double neutrons_per_second_per_cm() const { return this->neutrons_per_second() / this->spacing_dx(); }

    size_t cells_length() const { return m_cells_length; }
    double length_x() const { return m_length_x; }
    std::string absorption() const { return m_absorption; }
    std::string emission()   const { return m_emission; }
    std::string ucflux()     const { return m_ucflux; }
    double neutrons_per_second() const { return m_neutrons_per_second; }
    double absorption_per_cm() const { return m_absorption_per_cm; }

  private:
    size_t m_cells_length = 1;
    double m_length_x = 1.0;
    std::string m_absorption = "absorption";
    std::string m_emission = "emission";
    std::string m_ucflux = "ucflux";
    double m_neutrons_per_second = 1.0;
    double m_absorption_per_cm = 1.0;
};

TEST (dray_broomstick, dray_broomstick)
{
  Broomstick broomstick;
  broomstick.cells_length(3);
  broomstick.length_x(1.0);
  broomstick.absorption("absorption");
  broomstick.emission("emission");
  broomstick.ucflux("ucflux");
  broomstick.neutrons_per_second(1.0);
  broomstick.absorption_per_cm(1.0);

  conduit::Node data;
  const int legendre_order = 0;
  const int num_moments = (legendre_order+1) * (legendre_order+1);
  broomstick.set_up(data);
  
  /// std::shared_ptr<dray::UniformTopology> uni_topo
  ///     = dray::detail::import_topology_into_uniform(data, data["coordsets/coords"]);
  dray::DataSet dataset = dray::BlueprintReader::blueprint_to_dray_uniform(data);

  dray::FirstScatter integrator;
  integrator.legendre_order(legendre_order);
  integrator.total_cross_section_field("absorption");
  integrator.emission_field("emission");
  integrator.overwrite_first_scatter_field("ucflux");
  integrator.uniform_isotropic_scattering(0.0f);
  integrator.return_type(integrator.ReturnUncollidedFlux);
  /// integrator.falloff_none();
  integrator.execute(dataset);

  broomstick.check_ucflux_volume_src(dataset);
  fprintf(stdout, "\n");
  broomstick.check_ucflux_point_src(dataset);
}


// mimic kripke data
void Broomstick::set_up(conduit::Node &data)
{
  const size_t cells_width = 1;
  const size_t num_cells = this->cells_length() * cells_width * cells_width;

  // Pencil
  data["coordsets/coords/type"] = "uniform";
  data["coordsets/coords/dims/i"] = this->cells_length() + 1;
  data["coordsets/coords/dims/j"] = cells_width + 1;
  data["coordsets/coords/dims/k"] = cells_width + 1;

  // Position/spacing
  data["coordsets/coords/origin/x"] = 0.0;
  data["coordsets/coords/origin/y"] = 0.0;
  data["coordsets/coords/origin/z"] = 0.0;
  data["coordsets/coords/spacing/dx"] = this->spacing_dx();
  data["coordsets/coords/spacing/dy"] = 1.0;
  data["coordsets/coords/spacing/dz"] = 1.0;

  // Topology-->coordset
  data["topologies/topo/type"] = "uniform";
  data["topologies/topo/coordset"] = "coords";

  // Cell-centered fields, default 0.0:  absorption emission ucflux
  std::vector<double> dummy_field(num_cells, 0.0);
  conduit::Node &absorption_field = data["fields"][this->absorption()];
  absorption_field["association"] = "element";
  absorption_field["topology"] = "topo";
  absorption_field["values"].set(dummy_field.data(), num_cells);
  conduit::Node &emission_field = data["fields"][this->emission()];
  emission_field["association"] = "element";
  emission_field["topology"] = "topo";
  emission_field["values"].set(dummy_field.data(), num_cells);
  conduit::Node &ucflux_field = data["fields"][this->ucflux()];
  ucflux_field["association"] = "element";
  ucflux_field["topology"] = "topo";
  ucflux_field["values"].set(dummy_field.data(), num_cells);

  // Populate nonzeros of cell data
  double *emission= emission_field["values"].value();
  emission[0] = this->neutrons_per_second_per_cm();
  double *absorption= absorption_field["values"].value();
  for (size_t ii = 0; ii < num_cells; ++ii)
    absorption[ii] = this->absorption_per_cm();

  // Verify
  Node verify_info;
  if (!blueprint::mesh::verify(data, verify_info))
  {
    std::cout << "Mesh verify failed!\n";
    std::cout << verify_info.to_yaml() << "\n";
  }
  else
  {
    std::cout << "Mesh verify success!\n";
  }
}


void Broomstick::check_ucflux_volume_src(dray::DataSet &dataset)
{
  const dray::Float *ucflux = ((dray::LowOrderField *)
                               (dataset.field(this->ucflux())))
                              ->values().get_host_ptr_const();
  const double x0 = 0.0;
  const double x1 = this->spacing_dx();
  const double q = this->neutrons_per_second_per_cm();
  const double Sigma_t = this->absorption_per_cm();

  fprintf(stdout, "Volume source comparision.\n");
  fprintf(stdout, "%10s: ", "Expected");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double x = (0.5 + ii) * this->spacing_dx();
    const double expected = q / Sigma_t * (exp(-Sigma_t * (x - x1)) - exp(-Sigma_t * (x - x0)));
    fprintf(stdout, "%0.6e ", expected);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "%10s: ", "Actual");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double actual = ucflux[ii];
    fprintf(stdout, "%0.6e ", actual);
  }
  fprintf(stdout, "\n");
}

void Broomstick::check_ucflux_point_src(dray::DataSet &dataset)
{
  const dray::Float *ucflux = ((dray::LowOrderField *)
                               (dataset.field(this->ucflux())))
                              ->values().get_host_ptr_const();
  const double x0 = 0.0;
  const double x1 = this->spacing_dx();
  const double q = this->neutrons_per_second_per_cm();
  const double Sigma_t = this->absorption_per_cm();

  fprintf(stdout, "Point source comparision.\n");
  fprintf(stdout, "%10s: ", "Expected");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double x = (0.5 + ii) * this->spacing_dx();
    const double expected = q * (x1 - x0) / Sigma_t * (exp(-Sigma_t * (x - 0.5*(x0 + x1))));
    fprintf(stdout, "%0.6e ", expected);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "%10s: ", "Actual");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double actual = ucflux[ii];
    fprintf(stdout, "%0.6e ", actual);
  }
  fprintf(stdout, "\n");
}

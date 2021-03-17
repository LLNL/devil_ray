// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/io/blueprint_uniform_topology.hpp>
#include <dray/GridFunction/low_order_field.hpp>
#include <dray/spherical_harmonics.hpp>
#include <dray/filters/first_scatter.hpp>
#include <dray/array_utils.hpp>

#include <iostream>


class Broomstick
{
  public:
    void set_up(dray::DataSet &dataset, int legendre_order);
    void check_ucflux_volume_src(dray::DataSet &dataset, int legendre_order);
    void check_ucflux_point_src(dray::DataSet &dataset, int legendre_order);

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
  broomstick.cells_length(5);
  broomstick.length_x(1.0);
  broomstick.absorption("absorption");
  broomstick.emission("emission");
  broomstick.ucflux("ucflux");
  broomstick.neutrons_per_second(1.0);
  broomstick.absorption_per_cm(1.0);

  dray::DataSet dataset;
  const int legendre_order = 20;
  const int num_moments = (legendre_order+1) * (legendre_order+1);
  broomstick.set_up(dataset, legendre_order);

  dray::FirstScatter integrator;
  integrator.legendre_order(legendre_order);
  integrator.total_cross_section_field("absorption");
  integrator.emission_field("emission");
  integrator.overwrite_first_scatter_field("ucflux");
  integrator.uniform_isotropic_scattering(0.0f);
  integrator.return_type(integrator.ReturnUncollidedFlux);
  integrator.execute(dataset);

  broomstick.check_ucflux_volume_src(dataset, legendre_order);
  fprintf(stdout, "\n");
  broomstick.check_ucflux_point_src(dataset, legendre_order);
}


// mimic kripke data
void Broomstick::set_up(dray::DataSet &dataset, int legendre_order)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const size_t cells_width = 1;
  const size_t num_cells = this->cells_length() * cells_width * cells_width;
  const size_t num_moments = (legendre_order + 1) * (legendre_order + 1);

  using SH = dray::SphericalHarmonics<Float>;
  dray::SphericalHarmonics<Float> sh(legendre_order);
  constexpr dray::LowOrderField::Assoc Element = dray::LowOrderField::Assoc::Element;

  // Topology & spacing (pencil)
  const Vec<Float, 3> spacing = {{(Float) this->spacing_dx(), 1.0f, 1.0f}};
  const Float cell_volume = spacing[0] * spacing[1] * spacing[2];
  const Vec<Float, 3> origin = {{0, 0, 0}};
  const Vec<int32, 3> dims = {{(int32) this->cells_length(), cells_width, cells_width}};
  dataset = dray::DataSet(std::make_shared<dray::UniformTopology>(spacing, origin, dims));

  // Absorption field:  Isotropic Sigma_t
  dray::Array<Float> absorption_values;
  absorption_values.resize(num_cells);
  dray::array_memset_zero(absorption_values);
  for (int ii = 0; ii < num_cells; ++ii)
    absorption_values.get_host_ptr()[ii] = this->absorption_per_cm();
  std::shared_ptr<dray::LowOrderField> absorption_field
    = std::make_shared<dray::LowOrderField>(absorption_values, Element);
  absorption_field->name(this->absorption());
  dataset.add_field(absorption_field);

  // Emission field:  Anisotropic, project q(\hat{x})
  dray::Array<Float> emission_values;
  emission_values.resize(num_cells * num_moments);
  dray::array_memset_zero(emission_values);
  Float * cell_0_emission = emission_values.get_host_ptr() + (0 * num_moments);
  sh.project_delta(cell_0_emission, {{1, 0, 0}}, this->neutrons_per_second() / cell_volume);
  std::shared_ptr<dray::LowOrderField> emission_field
    = std::make_shared<dray::LowOrderField>(emission_values, Element);
  emission_field->name(this->emission());
  dataset.add_field(emission_field);

  // Uncollided flux field:   zeros
  dray::Array<Float> ucflux_values;
  ucflux_values.resize(num_cells * num_moments);
  dray::array_memset_zero(ucflux_values);
  std::shared_ptr<dray::LowOrderField> ucflux_field
    = std::make_shared<dray::LowOrderField>(ucflux_values, Element);
  ucflux_field->name(this->ucflux());
  dataset.add_field(ucflux_field);
}



double sqrt_4pi()
{
  const static double val = sqrt(4 * dray::pi());
  return val;
}

void Broomstick::check_ucflux_volume_src(dray::DataSet &dataset, int legendre_order)
{
  const dray::Float *ucflux = ((dray::LowOrderField *)
                               (dataset.field(this->ucflux())))
                              ->values().get_host_ptr_const();
  const double x0 = 0.0;
  const double x1 = this->spacing_dx();
  const double q = this->neutrons_per_second_per_cm();
  const double Sigma_t = this->absorption_per_cm();
  const size_t num_moments = (legendre_order + 1)*(legendre_order + 1);

  fprintf(stdout, "Volume source comparision.\n");
  fprintf(stdout, "%10s: ", "Expected");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double x = (0.5 + ii) * this->spacing_dx();
    const double expected = q / Sigma_t * (exp(-Sigma_t * (x - x1)) - exp(-Sigma_t * (x - x0)));
    fprintf(stdout, "%0.3e ", expected);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "%10s: ", "Actual");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double actual = sqrt_4pi() * ucflux[ii * num_moments + 0];
    fprintf(stdout, "%0.3e ", actual);
  }
  fprintf(stdout, "\n");
}

void Broomstick::check_ucflux_point_src(dray::DataSet &dataset, int legendre_order)
{
  const dray::Float *ucflux = ((dray::LowOrderField *)
                               (dataset.field(this->ucflux())))
                              ->values().get_host_ptr_const();
  const double x0 = 0.0;
  const double x1 = this->spacing_dx();
  const double q = this->neutrons_per_second_per_cm();
  const double Sigma_t = this->absorption_per_cm();
  const size_t num_moments = (legendre_order + 1)*(legendre_order + 1);

  fprintf(stdout, "Point source comparision.\n");
  fprintf(stdout, "%10s: ", "Expected");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double x = (0.5 + ii) * this->spacing_dx();
    const double expected = q * (x1 - x0) / Sigma_t * (exp(-Sigma_t * (x - 0.5*(x0 + x1))));
    fprintf(stdout, "%0.3e ", expected);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "%10s: ", "Actual");
  for (int ii = 1; ii < this->cells_length(); ++ii)
  {
    const double actual = sqrt_4pi() * ucflux[ii * num_moments + 0];
    fprintf(stdout, "%0.3e ", actual);
  }
  fprintf(stdout, "\n");
}

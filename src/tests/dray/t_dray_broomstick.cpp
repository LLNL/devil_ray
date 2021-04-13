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
#include <dray/uniform_faces.hpp>

#include <iostream>


class Broomstick
{
  public:
    void set_up(dray::DataSet &dataset, int legendre_order);
    void check_ucflux_volume_src(dray::DataSet &dataset, int legendre_order, int num_x_tail_sheets);
    void check_ucflux_point_src(dray::DataSet &dataset, int legendre_order);

    // properties
    void cell_dims(const dray::Vec<dray::int32, 3> &cell_dims) { m_cell_dims = cell_dims; }
    void length_xyz(const dray::Vec<dray::Float, 3> &length_xyz) { m_length_xyz = length_xyz; }
    void source_region_relative_length(const double src_x_fraction) { m_src_x_fraction = src_x_fraction; }
    void absorption(const std::string &absorption) { m_absorption = absorption; }
    void emission(const std::string &emission)     { m_emission = emission; }
    void ucflux(const std::string &ucflux)         { m_ucflux = ucflux; }
    void neutrons_per_second(double neutrons_per_second) { m_neutrons_per_second = neutrons_per_second; }
    void absorption_per_cm(double absorption_per_cm)    { m_absorption_per_cm = absorption_per_cm; }

    double spacing_dx() const { return this->length_xyz()[0] / this->cell_dims()[0]; }
    double spacing_dy() const { return this->length_xyz()[1] / this->cell_dims()[1]; }
    double spacing_dz() const { return this->length_xyz()[2] / this->cell_dims()[2]; }
    double spacing_dV() const { return spacing_dx() * spacing_dy() * spacing_dz(); }
    double source_volume() const { return length_xyz()[0] * source_region_relative_length()
                                        * length_xyz()[1] * length_xyz()[2]; }
    double neutrons_per_second_per_cm() const { return this->neutrons_per_second() / this->spacing_dx(); }
    double neutrons_per_second_per_cm3() const { return this->neutrons_per_second() / this->source_volume(); }

    dray::Vec<dray::int32, 3> cell_dims() const { return m_cell_dims; }
    dray::Vec<dray::Float, 3> length_xyz() const { return m_length_xyz; }
    double source_region_relative_length() const { return m_src_x_fraction; }
    std::string absorption() const { return m_absorption; }
    std::string emission()   const { return m_emission; }
    std::string ucflux()     const { return m_ucflux; }
    double neutrons_per_second() const { return m_neutrons_per_second; }
    double absorption_per_cm() const { return m_absorption_per_cm; }

  private:
    dray::Vec<dray::int32, 3> m_cell_dims = {{1, 1, 1}};
    dray::Vec<dray::Float, 3> m_length_xyz = {{1.0f, 1.0f, 1.0f}};
    double m_src_x_fraction = 1.0 / (1 << 20);
    std::string m_absorption = "absorption";
    std::string m_emission = "emission";
    std::string m_ucflux = "ucflux";
    double m_neutrons_per_second = 1.0;
    double m_absorption_per_cm = 1.0;
};


/*
TEST (dray_broomstick, dray_broomstick)
{
  Broomstick broomstick;
  const int short_dim = 1;
  const int long_dim = 1 << 8;
  broomstick.cell_dims({{long_dim, short_dim, short_dim}});
  broomstick.length_xyz({{long_dim*1.f, short_dim*1.f, short_dim*1.f}});
  broomstick.source_region_relative_length(short_dim*1.0 / long_dim);
  broomstick.absorption("absorption");
  broomstick.emission("emission");
  broomstick.ucflux("ucflux");
  broomstick.neutrons_per_second(1.0);
  broomstick.absorption_per_cm(1e-4);

  const int legendre_order = 20;

  fprintf(stdout, "-------------------------\n");
  fprintf(stdout, "cell_dims: [%d]x[%d]x[%d]\n",
      broomstick.cell_dims()[0], broomstick.cell_dims()[1], broomstick.cell_dims()[2]);
  fprintf(stdout, "phys_dims: [%e]x[%e]x[%e]\n",
      broomstick.length_xyz()[0], broomstick.length_xyz()[1], broomstick.length_xyz()[2]);
  fprintf(stdout, "neutrons/sec == %f\n", broomstick.neutrons_per_second());
  fprintf(stdout, "absorb/cm == %f\n", broomstick.absorption_per_cm());
  fprintf(stdout, "legendre_order == %d\n", legendre_order);
  fprintf(stdout, "-------------------------\n");

  dray::DataSet dataset;
  broomstick.set_up(dataset, legendre_order);

  dray::FirstScatter integrator;
  integrator.legendre_order(legendre_order);
  integrator.total_cross_section_field("absorption");
  integrator.emission_field("emission");
  integrator.overwrite_first_scatter_field("ucflux");
  integrator.uniform_isotropic_scattering(0.0f);
  integrator.return_type(integrator.ReturnUncollidedFlux);
  integrator.execute(dataset);

  broomstick.check_ucflux_volume_src(dataset, legendre_order, 5);
  fprintf(stdout, "\n");
  /// broomstick.check_ucflux_point_src(dataset, legendre_order);
}
*/


// mimic kripke data
void Broomstick::set_up(dray::DataSet &dataset, int legendre_order)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const size_t cells_width = 1;
  const Vec<int32, 3> cell_dims = this->cell_dims();
  const size_t num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];
  const size_t num_moments = (legendre_order + 1) * (legendre_order + 1);

  using SH = dray::SphericalHarmonics<Float>;
  dray::SphericalHarmonics<Float> sh(legendre_order);
  constexpr dray::LowOrderField::Assoc Element = dray::LowOrderField::Assoc::Element;

  // Topology & spacing (pencil)
  const Vec<Float, 3> spacing = {{(Float) this->spacing_dx(),
                                  (Float) this->spacing_dy(),
                                  (Float) this->spacing_dz()}};
  /// const Float cell_volume = this->spacing_dV();
  const Vec<Float, 3> origin = {{0, 0, 0}};
  dataset = dray::DataSet(std::make_shared<dray::UniformTopology>(spacing, origin, cell_dims));

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
  Float * emission_values_ptr = emission_values.get_host_ptr();
  const double src_rel_len = this->source_region_relative_length();
  for (int kk = 0; kk < cell_dims[2]; kk++)
    for (int jj = 0; jj < cell_dims[1]; jj++)
      for (int ii = 0; ii < cell_dims[0]
          && double(ii)/cell_dims[0] < src_rel_len; ++ii)
      {
        size_t cell_idx = ii + jj * cell_dims[0] + kk * cell_dims[0] * cell_dims[1];
        Float * cell_i_emission = emission_values_ptr + (cell_idx * num_moments);
        double dQ_dV = this->neutrons_per_second_per_cm3();
        if (!(double(ii + 1)/cell_dims[0] < src_rel_len))
          dQ_dV *= (src_rel_len - double(ii)/cell_dims[0]);
        sh.project_delta(cell_i_emission, {{1, 0, 0}}, dQ_dV);
      }
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

double four_pi()
{
  const static double val = 4 * dray::pi();
  return val;
}

void Broomstick::check_ucflux_volume_src(dray::DataSet &dataset, int legendre_order, int num_x_tail_sheets)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const Vec<int32, 3> dims = ((dray::UniformTopology*)dataset.topology())->cell_dims();
  const Vec<Float, 3> spacing = ((dray::UniformTopology*)dataset.topology())->spacing();
  const size_t num_cells = dims[0] * dims[1] * dims[2];

  const dray::Float *ucflux = ((dray::LowOrderField *)
                               (dataset.field(this->ucflux())))
                              ->values().get_host_ptr_const();
  const double x0 = 0.0;
  const double x1 = this->length_xyz()[0] * this->source_region_relative_length();
  const double q = this->neutrons_per_second_per_cm3();
  const double Sigma_t = this->absorption_per_cm();
  const double intern_transmit = 1.0 - exp(-Sigma_t * (x1 - x0));
  const size_t num_moments = (legendre_order + 1)*(legendre_order + 1);

  fprintf(stdout, "Broomstick volume source comparision (scalar flux).\n");
  fprintf(stdout, "%10s%10s%10s\n", "Cell", "Expected", "Actual");

  const int ii_begin = (dims[0] - num_x_tail_sheets > 0 ? dims[0] - num_x_tail_sheets : 0);
  for (int kk = 0; kk < dims[2]; ++kk)
    for (int jj = 0; jj < dims[1]; ++jj)
      for (int ii = ii_begin; ii < dims[0]; ++ii)
      {
        const double x = (0.5 + ii) * this->spacing_dx();
        if (x < x1)
          continue;

        const double expected_scalar_flux = q / Sigma_t * intern_transmit * exp(-Sigma_t * (x - x1));

        const int idx = ii + jj * dims[0] + kk * dims[0] * dims[1];
        const double actual_scalar_flux = sqrt_4pi() * ucflux[idx * num_moments + 0];

        fprintf(stdout, "%10d%10.3e%10.3e\n", idx, expected_scalar_flux, actual_scalar_flux);
      }
}

/// void Broomstick::check_ucflux_point_src(dray::DataSet &dataset, int legendre_order)
/// {
///   const dray::Float *ucflux = ((dray::LowOrderField *)
///                                (dataset.field(this->ucflux())))
///                               ->values().get_host_ptr_const();
///   const double x0 = 0.0;
///   const double x1 = this->spacing_dx();
///   const double q = this->neutrons_per_second_per_cm();
///   const double Sigma_t = this->absorption_per_cm();
///   const size_t num_moments = (legendre_order + 1)*(legendre_order + 1);
/// 
///   fprintf(stdout, "Point source comparision.\n");
///   fprintf(stdout, "%10s: ", "Expected");
///   for (int ii = 1; ii < this->cells_length(); ++ii)
///   {
///     const double x = (0.5 + ii) * this->spacing_dx();
///     const double expected = q * (x1 - x0) / Sigma_t * (exp(-Sigma_t * (x - 0.5*(x0 + x1))));
///     fprintf(stdout, "%0.3e ", expected);
///   }
///   fprintf(stdout, "\n");
///   fprintf(stdout, "%10s: ", "Actual");
///   for (int ii = 1; ii < this->cells_length(); ++ii)
///   {
///     const double actual = sqrt_4pi() * ucflux[ii * num_moments + 0];
///     fprintf(stdout, "%0.3e ", actual);
///   }
///   fprintf(stdout, "\n");
/// }



// ---------------------------------------------------------------


//
// uniform_dataset()
//
dray::DataSet uniform_dataset(const dray::Vec<dray::Float, 3> &spacing,
                              const dray::Vec<dray::Float, 3> &origin,
                              const dray::Vec<dray::int32, 3> &dims)
{
  return dray::DataSet(std::make_shared<dray::UniformTopology>(spacing, origin, dims));
}


//
// PointSource
//
class PointSource
{
  public:
    PointSource() = default;

    void legendre_order(int legendre_order) { m_legendre_order = legendre_order; }
    void source_cell(const dray::Vec<dray::int32, 3> &cell_idxs) { m_source_cell = cell_idxs; }
    void total_emission(double total_emission) { m_total_emission = total_emission; }
    void sigma_t(double sigma_t) { m_sigma_t = sigma_t; }

    int legendre_order() const { return m_legendre_order; }
    dray::Vec<dray::int32, 3> source_cell() const { return m_source_cell; }
    double total_emission() const { return m_total_emission; }
    double sigma_t() const { return m_sigma_t; }

    double analytical_pointwise_scalar_flux(const dray::Vec<dray::Float, 3> &src_x,
                                            const dray::Vec<dray::Float, 3> &x) const;

    void set_up(dray::DataSet &dataset,
                const std::string &emission,
                const std::string &absorption,
                const std::string &ucflux) const;

    void check_pointwise(dray::DataSet &dataset,
                         const std::string &ucflux) const;

    void check_cellavg(dray::DataSet &dataset,
                       const std::string &ucflux,
                       const dray::QuadratureRule & quadrature) const;

  private:
    int m_legendre_order = 0;
    dray::Vec<dray::int32, 3> m_source_cell = {{0, 0, 0}};
    double m_total_emission = 1;
    double m_sigma_t = 0;
};



TEST(dray_point_source, dray_point_source)
{
  int legendre_order = 4;
  int face_degree = 16;

  PointSource point_source;
  point_source.legendre_order(legendre_order);
  point_source.source_cell({{0, 0, 0}});
  point_source.total_emission(1.0);
  point_source.sigma_t(0.01);

  dray::DataSet dataset = uniform_dataset({{1.0/8, 1.0/8, 1.0/8}},
                                          {{0, 0, 0}},
                                          {{4, 4, 4}});

  point_source.set_up(dataset, "emission", "absorption", "ucflux");

  dray::FirstScatter integrator;
  integrator.legendre_order(legendre_order);
  integrator.face_quadrature_degree(face_degree);
  integrator.total_cross_section_field("absorption");
  integrator.emission_field("emission");
  integrator.overwrite_first_scatter_field("ucflux");
  integrator.uniform_isotropic_scattering(0.0f);
  integrator.return_type(integrator.ReturnUncollidedFlux);
  integrator.execute(dataset);

  /// point_source.check_pointwise(dataset, "ucflux");
  point_source.check_cellavg(dataset, "ucflux", dray::QuadratureRule::create(face_degree));
}


void PointSource::set_up(dray::DataSet &dataset,
                         const std::string &emission,
                         const std::string &absorption,
                         const std::string &ucflux) const
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const Vec<int32, 3> dims = ((dray::UniformTopology*)dataset.topology())->cell_dims();
  const Vec<Float, 3> spacing = ((dray::UniformTopology*)dataset.topology())->spacing();
  const size_t num_cells = dims[0] * dims[1] * dims[2];
  const Float cell_volume = spacing[0] * spacing[1] * spacing[2];
  const size_t legendre_order = this->legendre_order();
  const size_t num_moments = (legendre_order + 1) * (legendre_order + 1);

  using SH = dray::SphericalHarmonics<Float>;
  dray::SphericalHarmonics<Float> sh(legendre_order);
  constexpr dray::LowOrderField::Assoc Element = dray::LowOrderField::Assoc::Element;

  // Absorption field:  Isotropic Sigma_t
  dray::Array<Float> absorption_values;
  absorption_values.resize(num_cells);
  dray::array_memset_zero(absorption_values);
  for (int ii = 0; ii < num_cells; ++ii)
    absorption_values.get_host_ptr()[ii] = this->sigma_t();
  std::shared_ptr<dray::LowOrderField> absorption_field
    = std::make_shared<dray::LowOrderField>(absorption_values, Element);
  absorption_field->name(absorption);
  dataset.add_field(absorption_field);

  // Emission field:  Anisotropic, project q(\hat{x})
  dray::Array<Float> emission_values;
  emission_values.resize(num_cells * num_moments);
  dray::array_memset_zero(emission_values);
  const int32 cell_i = this->source_cell()[0]
                       + this->source_cell()[1] * dims[0]
                       + this->source_cell()[2] * dims[0] * dims[1];
  Float * cell_i_emission = emission_values.get_host_ptr() + (cell_i * num_moments);
  sh.project_isotropic(cell_i_emission, this->total_emission() / cell_volume);
  std::shared_ptr<dray::LowOrderField> emission_field
    = std::make_shared<dray::LowOrderField>(emission_values, Element);
  emission_field->name(emission);
  dataset.add_field(emission_field);

  // Uncollided flux field:   zeros
  dray::Array<Float> ucflux_values;
  ucflux_values.resize(num_cells * num_moments);
  dray::array_memset_zero(ucflux_values);
  std::shared_ptr<dray::LowOrderField> ucflux_field
    = std::make_shared<dray::LowOrderField>(ucflux_values, Element);
  ucflux_field->name(ucflux);
  dataset.add_field(ucflux_field);
}


double PointSource::analytical_pointwise_scalar_flux(const dray::Vec<dray::Float, 3> &src_x,
                                                     const dray::Vec<dray::Float, 3> &x) const
{
  const double Q = this->total_emission();
  const double Sigma_t = this->sigma_t();

  const double r2 = (x - src_x).magnitude2();
  const double r = sqrt(r2);
  return Q / (four_pi() * r2) * exp(-Sigma_t * r);
}


void PointSource::check_pointwise(dray::DataSet &dataset,
                                  const std::string &ucflux) const
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const Vec<int32, 3> dims = ((dray::UniformTopology*)dataset.topology())->cell_dims();
  const Vec<Float, 3> spacing = ((dray::UniformTopology*)dataset.topology())->spacing();
  const size_t num_cells = dims[0] * dims[1] * dims[2];

  const dray::Float *ucflux_values = ((dray::LowOrderField *)
                                      (dataset.field(ucflux)))
                                     ->values().get_host_ptr_const();

  const double Q = this->total_emission();
  const double Sigma_t = this->sigma_t();
  const size_t legendre_order = this->legendre_order();
  const size_t num_moments = (legendre_order + 1)*(legendre_order + 1);

  const int32 src_ii = this->source_cell()[0];
  const int32 src_jj = this->source_cell()[1];
  const int32 src_kk = this->source_cell()[2];

  const int32 cell_i = src_ii + src_jj * dims[0] + src_kk * dims[0] * dims[1];

  const Vec<Float, 3> src_x = {{ Float(src_ii + 0.5) * spacing[0],
                                 Float(src_jj + 0.5) * spacing[1],
                                 Float(src_kk + 0.5) * spacing[2] }};

  fprintf(stdout, "Isotropic point source comparision (scalar flux).\n");
  fprintf(stdout, "%10s%10s%10s%12s%10s\n", "Cell", "Expected", "Actual", "Rel. Err_0", "Err_linf");

  dray::SphericalHarmonics<Float> sh(legendre_order);

  for (int kk = 0; kk < dims[2]; ++kk)
    for (int jj = 0; jj < dims[1]; ++jj)
      for (int ii = 0; ii < dims[0]; ++ii)
      {
        const int idx = ii + jj * dims[0] + kk * dims[0] * dims[1];
        if (idx == cell_i)
          continue;

        const Vec<Float, 3> x = {{ Float(ii + 0.5) * spacing[0],
                                   Float(jj + 0.5) * spacing[1],
                                   Float(kk + 0.5) * spacing[2] }};

        const double expected_scalar_flux = this->analytical_pointwise_scalar_flux(src_x, x);
        const double actual_scalar_flux = sqrt_4pi() * ucflux_values[idx * num_moments + 0];
        const double diff_scalar_flux = abs(actual_scalar_flux - expected_scalar_flux);

        const Vec<Float, 3> normal = (x - src_x).normalized();
        const Float * sh_basis = sh.eval_all(normal);
        double diff_linf = 0.0;
        double diff_rel_linf = 0.0;
        for (int nm = 0; nm < num_moments; ++nm)
        {
          const double expected_moment = expected_scalar_flux * sh_basis[nm];
          const double actual_moment = ucflux_values[idx * num_moments + nm];
          const double diff = abs(actual_moment - expected_moment);
          const double diff_rel = diff / expected_moment;
          if (diff_linf < diff)
            diff_linf = diff;
          if (diff_rel_linf < diff_rel)
            diff_rel_linf = diff_rel;
        }

        fprintf(stdout, "%10d%10.3e%10.3e%12.0e%9.0f%%\n", idx,
            expected_scalar_flux, actual_scalar_flux,
            diff_scalar_flux / expected_scalar_flux,
            diff_rel_linf * 100.0);
      }
}


void PointSource::check_cellavg(dray::DataSet &dataset,
                                const std::string &ucflux,
                                const dray::QuadratureRule & quadrature) const
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  dray::UniformTopology * uni_topo = (dray::UniformTopology*)dataset.topology();
  const Vec<int32, 3> dims = uni_topo->cell_dims();
  const Vec<Float, 3> spacing = uni_topo->spacing();
  const size_t num_cells = dims[0] * dims[1] * dims[2];
  const Float cell_volume = spacing[0] * spacing[1] * spacing[2];

  const dray::Float *ucflux_values = ((dray::LowOrderField *)
                                      (dataset.field(ucflux)))
                                     ->values().get_host_ptr_const();

  const double Q = this->total_emission();
  const double Sigma_t = this->sigma_t();
  const size_t legendre_order = this->legendre_order();
  const size_t num_moments = (legendre_order + 1)*(legendre_order + 1);

  const int32 src_ii = this->source_cell()[0];
  const int32 src_jj = this->source_cell()[1];
  const int32 src_kk = this->source_cell()[2];

  const int32 cell_i = src_ii + src_jj * dims[0] + src_kk * dims[0] * dims[1];

  const Vec<Float, 3> src_x = {{ Float(src_ii + 0.5) * spacing[0],
                                 Float(src_jj + 0.5) * spacing[1],
                                 Float(src_kk + 0.5) * spacing[2] }};

  fprintf(stdout, "Isotropic point source comparision (scalar flux).\n");
  fprintf(stdout, "%10s%10s%10s%12s%10s\n", "Cell", "Expected", "Actual", "Rel. Err_0", "Err_linf");

  const dray::UniformFaces face_map = dray::UniformFaces::from_uniform_topo(*uni_topo);

  using FaceID = dray::UniformFaces::FaceID;

  dray::Array<Vec<Float, 3>> face_points;
  dray::Array<Float> face_weights;
  const int points_per_face = quadrature.points() * quadrature.points();
  const int points_per_cell = FaceID::NUM_FACES * points_per_face;
  face_points.resize(face_map.num_total_faces() * points_per_face);
  face_weights.resize(face_map.num_total_faces() * points_per_face);
  face_map.fill_total_faces(face_points.get_host_ptr(),
                            face_weights.get_host_ptr(),
                            quadrature);
  const Vec<Float, 3> * face_points_p = face_points.get_host_ptr_const();
  const Float * face_weights_p = face_weights.get_host_ptr_const();

  dray::SphericalHarmonics<Float> sh(legendre_order);

  for (int kk = 0; kk < dims[2]; ++kk)
    for (int jj = 0; jj < dims[1]; ++jj)
      for (int ii = 0; ii < dims[0]; ++ii)
      {
        const int idx = ii + jj * dims[0] + kk * dims[0] * dims[1];
        if (idx == cell_i)
          continue;

        const Vec<Float, 3> x = {{ Float(ii + 0.5) * spacing[0],
                                   Float(jj + 0.5) * spacing[1],
                                   Float(kk + 0.5) * spacing[2] }};
        const Float r2 = (x - src_x).magnitude2();
        const Float r = sqrt(r2);

        std::vector<double> expected_scalar_flux_pointwise_faces(points_per_cell);
        std::vector<Float> face_weights(points_per_cell);
        Vec<Float, 3> omega_faces[points_per_cell];
        Vec<Float, 3> omega_hat_faces[points_per_cell];
        for (dray::uint8 f = 0; f < FaceID::NUM_FACES; ++f)
          for (int quad_idx = 0; quad_idx < points_per_face; ++quad_idx)
          {
            const FaceID face = FaceID(f);
            const int32 face_idx = face_map.cell_idx_to_face_idx(idx, face) * points_per_face + quad_idx;
            const Vec<Float, 3> face_x = face_points_p[face_idx];
            face_weights[f * points_per_face + quad_idx] = face_weights_p[face_idx];

            expected_scalar_flux_pointwise_faces[f * points_per_face + quad_idx] =
                this->analytical_pointwise_scalar_flux(src_x, face_x);

            omega_faces[f * points_per_face + quad_idx] = (face_x - src_x);
            omega_hat_faces[f * points_per_face + quad_idx] = (face_x - src_x).normalized();
          }

        /// for (int f = 0; f < 6; ++f)
        /// {
        ///   double r_0 = omega_faces[0].magnitude();
        ///   double r_f = omega_faces[f].magnitude();
        ///   fprintf(stdout, " (%s)", (abs(r_f - r_0) < 1e-2 ? "=" : r_f < r_0 ? "<" : ">"));
        /// }
        /// fprintf(stdout, "\n");
        /// for (int f = 0; f < 6; ++f)
        /// {
        ///   /// fprintf(stdout, " %.1f", expected_scalar_flux_pointwise_faces[f]);
        ///   double v_0 = expected_scalar_flux_pointwise_faces[0];
        ///   double v_f = expected_scalar_flux_pointwise_faces[f];
        ///   fprintf(stdout, " (%s)", (abs(v_f - v_0) < 1e-2 ? "=" : v_f < v_0 ? "<" : ">"));
        /// }
        /// fprintf(stdout, "\n");
        /// fprintf(stdout, "\n");

        double expected_scalar_flux_cellavg = 0.0f;
        for (dray::uint8 f = 0; f < FaceID::NUM_FACES; ++f)
          for (int quad_idx = 0; quad_idx < points_per_face; ++quad_idx)
          {
            const double face_cosine =
                dot(face_map.normal(FaceID(f)), omega_hat_faces[f * points_per_face + quad_idx]);
            const double face_area = face_map.face_area(FaceID(f));
            const double weight = face_weights[f * points_per_face + quad_idx];

            expected_scalar_flux_cellavg +=
              - expected_scalar_flux_pointwise_faces[f * points_per_face + quad_idx]
                * face_area
                * face_cosine
                * weight
                / (Sigma_t * cell_volume);
          }
        const bool is_negative = (expected_scalar_flux_cellavg < 0.0f);

        /*
        if (is_negative)
        {
          double sum = 0.0f;
          for (dray::uint8 f = 0; f < FaceID::NUM_FACES; ++f)
          {
            const double face_cosine =
                dot(face_map.normal(FaceID(f)), omega_hat_faces[f]);
            /// const double face_area = face_map.face_area(FaceID(f));

            fprintf(stdout, " -(%.1f)(%.1f)",
                expected_scalar_flux_pointwise_faces[f],
                face_cosine);

            sum += - expected_scalar_flux_pointwise_faces[f] * face_cosine;
          }
          fprintf(stdout, "= %.4f\n", sum);
        }
        */

        const double actual_scalar_flux = sqrt_4pi() * ucflux_values[idx * num_moments + 0];
        const double diff_scalar_flux = abs(actual_scalar_flux - expected_scalar_flux_cellavg);

        /*
        const Vec<Float, 3> normal = (x - src_x).normalized();
        const Float * sh_basis = sh.eval_all(normal);
        double diff_linf = 0.0;
        double diff_rel_linf = 0.0;
        for (int nm = 0; nm < num_moments; ++nm)
        {
          //TODO have to multiply by the sh at the face level
          const double expected_moment = expected_scalar_flux_cellavg * sh_basis[nm];
          const double actual_moment = ucflux_values[idx * num_moments + nm];
          const double diff = abs(actual_moment - expected_moment);
          const double diff_rel = diff / expected_moment;
          if (diff_linf < diff)
            diff_linf = diff;
          if (diff_rel_linf < diff_rel)
            diff_rel_linf = diff_rel;
        }
        */

        /// fprintf(stdout, "%10d%10.3e%10.3e%12.0e%9.0f%%\n", idx,
        fprintf(stdout, "%10d%10.3e%10.3e%12.0e\n", idx,
            expected_scalar_flux_cellavg, actual_scalar_flux,
            diff_scalar_flux / expected_scalar_flux_cellavg//,
            //diff_rel_linf * 100.0);
          );
      }
}


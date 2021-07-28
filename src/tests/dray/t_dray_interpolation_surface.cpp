// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// test
#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

// basic dray
#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>
#include <dray/array_utils.hpp>
#include <dray/ray.hpp>

// dataset
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/data_model/low_order_field.hpp>

// rendering
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/device_framebuffer.hpp>

// new data structures
#include <dray/quadtree.hpp>
#include <dray/interpolation_surface.hpp>

// io
/// #include <conduit_blueprint.hpp>
/// #include <conduit_relay.hpp>
#include <iostream>


class OpaqueBlocker
{
  private:
    dray::Vec<dray::Float, 3> m_min;
    dray::Vec<dray::Float, 3> m_max;
    DRAY_EXEC OpaqueBlocker();
  public:
    DRAY_EXEC OpaqueBlocker( const dray::Vec<dray::Float, 3> &min,
                             const dray::Vec<dray::Float, 3> &max);
    DRAY_EXEC OpaqueBlocker(const OpaqueBlocker &) = default;
    DRAY_EXEC OpaqueBlocker(OpaqueBlocker &&) = default;
    DRAY_EXEC OpaqueBlocker & operator=(const OpaqueBlocker &) = default;
    DRAY_EXEC OpaqueBlocker & operator=(OpaqueBlocker &&) = default;
    DRAY_EXEC bool visibility(const dray::Ray &ray) const;
};

template <int dim_I, int dim_J, typename F>
dray::Float integrate_plane(
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 2> &sides,
    F integrand,
    double rel_tol);

template <int dim_I, int dim_J, typename F>
void crazy_uniform(
    const dray::Vec<double, 3> &plane_center,
    const dray::Vec<double, 2> &sides,
    F integrand,
    int num_levels,
    double * level_results);

template <int dim_I, int dim_J, typename F>
int adaptive_trapezoid(
    const dray::Vec<double, 3> &plane_center,
    const dray::Vec<double, 2> &sides,
    F integrand,
    int min_levels,
    int num_levels,
    double rel_tol,
    double * level_results);

//
// dray_interpolation_surface
//
TEST (dray_interpolation_surface, dray_basic)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const Vec<Float, 3> source = {{0, 0.5f, 0.5f}};
  const Float strength = 1;
  const Float sigma_0 = 0.0f;
  const Float sigma_half = log(2);
  const Float sigma_max = 256.f;

  // Mesh: 1x1x1 + 1x1x1

  // .25 x .1 x .1 occluder
  OpaqueBlocker blocker({{0.50f, 0.45f, 0.45}},
                        {{0.75f, 0.55f, 0.55}});

  const Float shadow_area_0 = (1.0 / 0.50) * (0.55-0.45)
                            * (1.0 / 0.50) * (0.55-0.45);

  // Function on inter-domain surface resulting from domain 1:
  const auto sigma_visible = [&](const Vec<double, 3> &x) {
    /// return sigma_0;
    /// return sigma_half;

    dray::Ray ray;
    ray.m_dir = x - source;
    ray.m_orig = source;
    return blocker.visibility(ray) ? sigma_0 : sigma_max;
  };

  // true flux using sigma_visible
  const auto flux = [&](const Vec<double, 3> &x,
                        const Vec<double, 3> &normal) {
    const Vec<double, 3> r = x - source;
    const double r2 = r.magnitude2(),  r1 = sqrt(r2);
    return strength *
           dray::rcp_safe(r2) *
           exp(-sigma_visible(x) * r1) *
           dot(r.normalized(), normal);
  };


  // Ground truth:
  //   AdaptiveTrapezoid -- Integrate current on all faces, should sum to 0.
  //                     -- Use double precision.
  //                     -- Adjust rel_tol, num_levels, and min_levels.
  const int num_planes = 6;

  const Vec<double, 3> plane_centers[num_planes] = {
    {{1.0, 0.5, 0.5}},
    {{2.0, 0.5, 0.5}},
    {{1.5, 0.0, 0.5}},
    {{1.5, 1.0, 0.5}},
    {{1.5, 0.5, 0.0}},
    {{1.5, 0.5, 1.0}}
  };
  const int plane_sides[num_planes] = {1, 1, 1, 1, 1, 1};

  const int min_levels = 2;
  const int num_levels = 24;
  const double rel_tol = 1e-6;

  // --------------------------------------------------
  // Array of (array of result per level) per plane
  double integrations[num_planes][num_levels];

  // integrate
  const Vec<double, 2> one_one = {{1, 1}};
  std::cout << "levels used:  ";
  std::cout << "\t" << adaptive_trapezoid<1, 2>( plane_centers[0], one_one * plane_sides[0],
                       flux, min_levels, num_levels, rel_tol, integrations[0] );
  std::cout << "\t" << adaptive_trapezoid<1, 2>( plane_centers[1], one_one * plane_sides[1],
                       flux, min_levels, num_levels, rel_tol, integrations[1] );
  std::cout << "\t" << adaptive_trapezoid<0, 2>( plane_centers[2], one_one * plane_sides[2],
                       flux, min_levels, num_levels, rel_tol, integrations[2] );
  std::cout << "\t" << adaptive_trapezoid<0, 2>( plane_centers[3], one_one * plane_sides[3],
                       flux, min_levels, num_levels, rel_tol, integrations[3] );
  std::cout << "\t" << adaptive_trapezoid<0, 1>( plane_centers[4], one_one * plane_sides[4],
                       flux, min_levels, num_levels, rel_tol, integrations[4] );
  std::cout << "\t" << adaptive_trapezoid<0, 1>( plane_centers[5], one_one * plane_sides[5],
                       flux, min_levels, num_levels, rel_tol, integrations[5] );
  std::cout << "\n";

  const int print_planes = 6;

  // print
  std::cout << "plane:" << std::right;
  for (int plane_idx = 0; plane_idx < print_planes; ++plane_idx) std::cout << " \t" << plane_idx;
  std::cout << "\n";
  for (int level = 0; level < num_levels; ++level)
  {
    std::cout << " level=" << level;
    for (int plane_idx = 0; plane_idx < print_planes; ++plane_idx)
      std::cout << " \t" << std::setprecision(10) << std::fixed << integrations[plane_idx][level];
    std::cout << "\n";
  }
  std::cout << std::left;

  const double current_in = abs(integrations[0][num_levels-1]);
  const double current_out =
      abs(integrations[1][num_levels-1]) +
      abs(integrations[2][num_levels-1]) +
      abs(integrations[3][num_levels-1]) +
      abs(integrations[4][num_levels-1]) +
      abs(integrations[5][num_levels-1]);

  std::cout << "-----------------------------\n";
  std::cout << "Current In  == " << current_in << "\n";
  std::cout << "Current Out == " << current_out << "\n";
  std::cout << "Current In - Out == "
            << std::setprecision(3) << std::scientific
            << current_in - current_out << "\n";


  // Store samples of {\bar{Simga_t}} on interpolation surface.
  //   -- until error of integrating {\bar{Sigma_t} dA} meets threshold

  // Approximate currents in domain 1 (far domain)
  //   -- (Adaptive or uniform) quadrature
  //   -- Intersect samples at the interpolation surface
  //   -- Reduce


}


template <int dims, int axis, typename T>
dray::Vec<T, dims> axis_vec(T length = 1.0)
{
  dray::Vec<T, dims> v;
  v = 0;
  v[axis] = length;
  return v;
}



template <int dim_I, int dim_J, typename F>
void crazy_uniform(
    const dray::Vec<double, 3> &plane_center,
    const dray::Vec<double, 2> &sides,
    F integrand,
    int num_levels,
    double * level_results)
{
  using namespace dray;

  static std::vector<double> total_aux;
  const size_t restore_aux_size = total_aux.size();
  total_aux.reserve(num_levels * (num_levels-1) / 2);
  total_aux.resize(restore_aux_size + num_levels - 1);

  level_results[0] = integrand(plane_center) * sides[0] * sides[1];

  if (num_levels > 1)
  {
    for (int level = 0; level < num_levels - 1; ++level)
      level_results[level + 1] = 0;
    for (int child = 0; child < 4; ++child)
    {
      const int child_i = ((child >> 0) & 1u) * 2 - 1;
      const int child_j = ((child >> 1) & 1u) * 2 - 1;

      crazy_uniform<dim_I, dim_J>(
          plane_center +
            axis_vec<3, dim_I>(sides[0] * child_i / 4) +
            axis_vec<3, dim_J>(sides[1] * child_j / 4),
          sides / 2,
          integrand,
          num_levels - 1,
          &total_aux[restore_aux_size]);

      for (int level = 0; level < num_levels - 1; ++level)
        level_results[level + 1] += (&total_aux[restore_aux_size])[level];
    }
  }

  total_aux.resize(restore_aux_size);
}


template <int keep_bits>
double truncate(double number)
{
  const int low = 52 - keep_bits;
  unsigned long long bits = *(unsigned long long *) &number;
  bits &= ~((1llu << low) - 1);
  return *(double *) &bits;
}


template <int dim_I, int dim_J, typename F>
int adaptive_trapezoid(
    const dray::Vec<double, 3> &plane_center,
    const dray::Vec<double, 2> &sides,
    F integrand,
    int min_levels,
    int num_levels,
    double rel_tol,
    double * level_results)
{
  using namespace dray;

  static std::vector<double> total_aux;
  const size_t restore_aux_size = total_aux.size();
  total_aux.reserve(num_levels * (num_levels-1) / 2);
  total_aux.resize(restore_aux_size + num_levels - 1);

  static const Vec<double, 3> normal =
      axis_vec<3, (2*dim_J - dim_I + 3) % 3>(
          ((dim_J - dim_I + 1) % 3) - 1.0);

  const double dA = sides[0] * sides[1];

  const double sample_mid = integrand(plane_center, normal);

  double sample_corner[4];
  for (int child = 0; child < 4; ++child)
  {
    const int child_i = ((child >> 0) & 1u) * 2 - 1;
    const int child_j = ((child >> 1) & 1u) * 2 - 1;

    sample_corner[child] = integrand(
        plane_center +
          axis_vec<3, dim_I>(sides[0] * child_i / 2) +
          axis_vec<3, dim_J>(sides[1] * child_j / 2),
        normal );
  }
  const double interp_mid = 0.25 * (
      sample_corner[0] + sample_corner[1] + sample_corner[2] + sample_corner[3] );

  level_results[0] = interp_mid * dA;

  int sub_levels_used = 0;

  if (num_levels > 1)
  {
    const double rel_err = (interp_mid == sample_mid ? 0 :
        abs(interp_mid - sample_mid) / (0.5 * (abs(interp_mid) + abs(sample_mid))));

    if (min_levels <= 1 && rel_err < rel_tol)  // If accurate, do not recurse
    {
      for (int level = 0; level < num_levels - 1; ++level)
        level_results[level + 1] = interp_mid * dA;
    }

    else  // Need accuracy by recursion
    {
      for (int level = 0; level < num_levels - 1; ++level)
        level_results[level + 1] = 0;
      for (int child = 0; child < 4; ++child)
      {
        const int child_i = ((child >> 0) & 1u) * 2 - 1;
        const int child_j = ((child >> 1) & 1u) * 2 - 1;

        int child_levels_used =
            adaptive_trapezoid<dim_I, dim_J>(
                plane_center +
                  axis_vec<3, dim_I>(sides[0] * child_i / 4) +
                  axis_vec<3, dim_J>(sides[1] * child_j / 4),
                sides / 2,
                integrand,
                min_levels - 1,
                num_levels - 1,
                rel_tol,
                &total_aux[restore_aux_size]);

        for (int level = 0; level < num_levels - 1; ++level)
          level_results[level + 1] += (&total_aux[restore_aux_size])[level];

        if (sub_levels_used < child_levels_used)
          sub_levels_used = child_levels_used;
      }
    }
  }

  total_aux.resize(restore_aux_size);

  return sub_levels_used + 1;
}





// OpaqueBlocker()
DRAY_EXEC OpaqueBlocker::OpaqueBlocker()
  : m_min({{0, 0, 0}}),
    m_max({{1, 1, 1}})
{}

// OpaqueBlocker()
DRAY_EXEC OpaqueBlocker::OpaqueBlocker(
  const dray::Vec<dray::Float, 3> &min,
  const dray::Vec<dray::Float, 3> &max)
  : m_min(min),
    m_max(max)
{}

// OpaqueBlocker::visibility()
DRAY_EXEC bool OpaqueBlocker::visibility(const dray::Ray &ray) const
{
  dray::Range t_range = dray::Range::mult_identity();  // intesections
  for (dray::int32 d = 0; d < 3; ++d)
  {
    const dray::Float t_0 = (m_min[d] - ray.m_orig[d]) / ray.m_dir[d];
    const dray::Float t_1 = (m_max[d] - ray.m_orig[d]) / ray.m_dir[d];
    dray::Range range_i = dray::Range::identity();  // unions
    range_i.include(t_0);
    range_i.include(t_1);
    t_range = t_range.intersect(range_i);
  }
  return t_range.is_empty();
}




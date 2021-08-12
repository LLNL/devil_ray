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
#include <dray/array.hpp>
#include <dray/device_array.hpp>
#include <dray/array_utils.hpp>
#include <dray/ray.hpp>
#include <dray/matrix.hpp>

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



struct Reconstructor
{
  dray::Vec<double, 3> m_origin;
  dray::Vec<double, 3> m_delta_a;
  dray::Vec<double, 3> m_delta_b;
  double m_kappa = 1.0;

  struct Quad
  {
    dray::Vec<double, 2> m_lower_left;
    double m_side;
    size_t m_begin_children = -1;

    Quad() : m_lower_left({{0, 0}}), m_side(1) { }
    Quad(const Quad &) = default;
    Quad & operator=(const Quad &) = default;
    dray::Vec<double, 2> center() const
    {
      return m_lower_left + dray::Vec<double, 2>{{0.5, 0.5}} * m_side;
    }
    double side() const { return m_side; }
    Quad child(int child_num) const
    {
      Quad child;
      child.m_lower_left = m_lower_left;
      child.m_lower_left[0] += m_side * (0.5 * ((child_num >> 0) & 1u));
      child.m_lower_left[1] += m_side * (0.5 * ((child_num >> 1) & 1u));
      child.m_side = m_side * 0.5;
      return child;
    }
    bool is_leaf() const { return m_begin_children == -1; }
    void begin_children(size_t begin_children) { m_begin_children = begin_children; }
    size_t begin_children() const { return m_begin_children; }
  };

  std::vector<Quad> m_quads;

  // construct
  Reconstructor(
      const dray::Vec<double, 3> &origin,
      const dray::Vec<double, 3> &delta_a,
      const dray::Vec<double, 3> &delta_b,
      int starting_leaf_level);

  // prop
  void kappa(double kappa) { m_kappa = kappa; }
  double kappa() const { return m_kappa; }

  size_t size() const { return m_quads.size(); }

  // bulk
  template <typename EvalTol>
  bool improve_resolution(const EvalTol & eval_tol);

  template <typename Data, typename EvalFunction>
  void store_samples(
      const EvalFunction & eval_func,
      dray::Array<Data> & rtn_samples);

  // pinpoint
  template <typename Data>
  Data interpolate(
      const dray::ConstDeviceArray<Data> &samples,
      const dray::Vec<double, 3> &x) const;
};




//
// dray_interpolation_surface
//
TEST (dray_interpolation_surface, dray_basic)
{
  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const Vec<Float, 3> source = {{0.0f, 0.5f, 0.5f}};
  /// const Vec<Float, 3> source = {{0.375f, 0.5f, 0.5f}};
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

  const auto flux_aligned = [&](const Vec<double, 3> &x) {
    const Vec<double, 3> r = x - source;
    const double r2 = r.magnitude2(),  r1 = sqrt(r2);
    return strength *
           dray::rcp_safe(r2) *
           exp(-sigma_visible(x) * r1);
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
  const int num_levels = 18;
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

  std::cout << std::flush;

  // Store samples of {\bar{Simga_t}} on interpolation surface.
  //   -- until error of integrating {\bar{Sigma_t} dA} meets threshold

  Reconstructor path_length(
      {{1.0, 0.0, 0.0}},  // origin
      {{0.0, 1.0, 0.0}},  // y side
      {{0.0, 0.0, 1.0}},
      1); // z side
  path_length.kappa(1e0);

  const double abs_tau_flux = 1e-3;

  const auto eval_tol = [&](const Vec<double, 3> &x, double radius)
  {
    const double half_side = radius;

    Vec<double, 3> x00 = x, x01 = x, x10 = x, x11 = x;
    x00[1] -= half_side;  x00[2] -= half_side;
    x01[1] += half_side;  x01[2] -= half_side;
    x10[1] -= half_side;  x10[2] += half_side;
    x11[1] += half_side;  x11[2] += half_side;

    const double max_flux_mag =
        max(
            max(flux_aligned(x00), flux_aligned(x01)),
            max(flux_aligned(x10), flux_aligned(x11))
           );

    return abs_tau_flux / max_flux_mag;
  };

  const auto path_length_function = [&](const Vec<double, 3> &x)
  {
    const double dist1 = (x - source).magnitude();
    return sigma_visible(x) * dist1;
  };

  bool changed = false;
  do
  {
    changed = path_length.improve_resolution(eval_tol);

    std::cout
      << "abs_tau_flux == " << abs_tau_flux
      << "    size == " << path_length.size() << "\n";
  }
  while (changed);

  dray::Array<double> plength_samples;
  path_length.store_samples(path_length_function, plength_samples);
  dray::ConstDeviceArray<double> d_plength_samples(plength_samples);

  const auto sigma_interpolated = [&](const Vec<double, 3> &x) {
    Vec<double, 3> dir = x - source;
    Vec<double, 3> intercept = source + dir * (1.0-source[0]) * dray::rcp_safe(dir[0]);
    return path_length.interpolate(d_plength_samples, intercept);
  };

  // flux using sigma interpolated on interpolation surface.
  const auto flux_apprx_sigma = [&](const Vec<double, 3> &x,
                        const Vec<double, 3> &normal) {
    const Vec<double, 3> r = x - source;
    const double r2 = r.magnitude2(),  r1 = sqrt(r2);
    return strength *
           dray::rcp_safe(r2) *
           exp(-sigma_interpolated(x) * r1) *
           dot(r.normalized(), normal);
  };

  std::cout << "-----------------------------\n";
  std::cout << "(apprx) levels used";

  double apprx_integrations[num_planes][num_levels];

  std::cout << "\t" << adaptive_trapezoid<1, 2>( plane_centers[0], one_one * plane_sides[0],
      flux_apprx_sigma, min_levels, num_levels, rel_tol, apprx_integrations[0] );
  std::cout << "\t" << adaptive_trapezoid<1, 2>( plane_centers[1], one_one * plane_sides[1],
      flux_apprx_sigma, min_levels, num_levels, rel_tol, apprx_integrations[1] );
  std::cout << "\t" << adaptive_trapezoid<0, 2>( plane_centers[2], one_one * plane_sides[2],
      flux_apprx_sigma, min_levels, num_levels, rel_tol, apprx_integrations[2] );
  std::cout << "\t" << adaptive_trapezoid<0, 2>( plane_centers[3], one_one * plane_sides[3],
      flux_apprx_sigma, min_levels, num_levels, rel_tol, apprx_integrations[3] );
  std::cout << "\t" << adaptive_trapezoid<0, 1>( plane_centers[4], one_one * plane_sides[4],
      flux_apprx_sigma, min_levels, num_levels, rel_tol, apprx_integrations[4] );
  std::cout << "\t" << adaptive_trapezoid<0, 1>( plane_centers[5], one_one * plane_sides[5],
      flux_apprx_sigma, min_levels, num_levels, rel_tol, apprx_integrations[5] );

  std::cout << "\n";

  for (int plane_idx = 0; plane_idx < print_planes; ++plane_idx)
    std::cout << " \t" << std::setprecision(10) << std::fixed << apprx_integrations[plane_idx][num_levels-1];
  std::cout << "\n";
  std::cout << std::left;

  const double apprx_current_in = abs(apprx_integrations[0][num_levels-1]);
  const double apprx_current_out =
      abs(apprx_integrations[1][num_levels-1]) +
      abs(apprx_integrations[2][num_levels-1]) +
      abs(apprx_integrations[3][num_levels-1]) +
      abs(apprx_integrations[4][num_levels-1]) +
      abs(apprx_integrations[5][num_levels-1]);

  std::cout << "-----------------------------\n";
  std::cout << "(apprx) Current In  == " << apprx_current_in << "\n";
  std::cout << "(apprx) Current Out == " << apprx_current_out << "\n";
  std::cout << "(apprx) Current In - Out == "
            << std::setprecision(3) << std::scientific
            << apprx_current_in - apprx_current_out << "\n";
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




void complete_tree(
    const typename Reconstructor::Quad &root,
    int height,
    std::vector<typename Reconstructor::Quad> &node_sink,
    size_t root_idx)
{
  if (height > 0)
  {
    const size_t begin_children = node_sink.size();
    node_sink[root_idx].begin_children(begin_children);
    for (int child = 0; child < 4; ++child)
      node_sink.push_back(root.child(child));
    for (int child = 0; child < 4; ++child)
      complete_tree(
          root.child(child), height-1, node_sink, begin_children + child);
  }
}


Reconstructor::Reconstructor(
    const dray::Vec<double, 3> &origin,
    const dray::Vec<double, 3> &delta_a,
    const dray::Vec<double, 3> &delta_b,
    int starting_leaf_level)
  :
    m_origin(origin), m_delta_a(delta_a), m_delta_b(delta_b)
{
  /// m_err.push_back(dray::infinity<double>());
  m_quads.push_back(Quad());
  complete_tree(m_quads[0], starting_leaf_level, m_quads, 0);
}

template <typename EvalTol>
bool Reconstructor::improve_resolution(const EvalTol & eval_tol)
{
  const size_t in_tree_size = m_quads.size();

  bool subdivided = false;

  for (size_t in_node = 0; in_node < in_tree_size; ++in_node)
  {
    const double side = m_quads[in_node].side();
    const dray::Vec<double, 2> center = m_quads[in_node].center();
    const dray::Vec<double, 3> wcenter =
        m_origin + m_delta_a * center[0] + m_delta_b * center[1];
    const double radius = (m_delta_a.magnitude() + m_delta_b.magnitude()) * 0.5;

    const bool is_leaf = m_quads[in_node].is_leaf();
    const double evald_err = eval_tol(wcenter, radius);

    if (is_leaf && side > this->kappa() * evald_err)
    {
      // subdivide
      m_quads[in_node].begin_children(m_quads.size());
      m_quads.push_back(m_quads[in_node].child(0));
      m_quads.push_back(m_quads[in_node].child(1));
      m_quads.push_back(m_quads[in_node].child(2));
      m_quads.push_back(m_quads[in_node].child(3));

      subdivided = true;
    }
  }

  return subdivided;
}

template <typename Data, typename EvalFunction>
void Reconstructor::store_samples(
    const EvalFunction & eval_func,
    dray::Array<Data> & rtn_samples)
{
  const Reconstructor * reconstructor = this;
  const dray::Vec<double, 3> o = m_origin, a = m_delta_a, b = m_delta_b;

  rtn_samples.resize(4 * this->size());
  dray::NonConstDeviceArray<Data> d_rtn_samples(rtn_samples);
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, this->size()),
      [&](int ii)
  {
    if (reconstructor->m_quads[ii].is_leaf())
    {
      const double h = reconstructor->m_quads[ii].side() / 2;
      const dray::Vec<double, 2> c = reconstructor->m_quads[ii].center();
      d_rtn_samples.get_item(4*ii+0) = 0.25 * eval_func(o + a * (c[0] - h) + b * (c[1] - h));
      d_rtn_samples.get_item(4*ii+1) = 0.25 * eval_func(o + a * (c[0] + h) + b * (c[1] - h));
      d_rtn_samples.get_item(4*ii+2) = 0.25 * eval_func(o + a * (c[0] - h) + b * (c[1] + h));
      d_rtn_samples.get_item(4*ii+3) = 0.25 * eval_func(o + a * (c[0] + h) + b * (c[1] + h));
    }
  });
}


template <typename T>
struct PlaneProject
{
  dray::Vec<T, 3> m_da;
  dray::Vec<T, 3> m_db;
  dray::Matrix<T, 2, 2> m_g_inv;

  PlaneProject(const dray::Vec<T, 3> &da,
               const dray::Vec<T, 3> &db)
    : m_da(da), m_db(db)
  {
    dray::Matrix<T, 2, 2> g;
    g(0, 0) = dray::dot(da, da);
    g(0, 1) = dray::dot(da, db);
    g(1, 0) = g(0, 1);
    g(1, 1) = dray::dot(db, db);
    bool valid;
    m_g_inv = dray::matrix_inverse(g, valid);
    // If not valid, then plane is degenerate.
  }

  dray::Vec<T, 2> project(const dray::Vec<T, 3> &w) const
  {
    return m_g_inv * dray::Vec<T, 2>{{ dray::dot(w, m_da),
                                       dray::dot(w, m_db) }};
  }
};




// pinpoint
template <typename Data>
Data Reconstructor::interpolate(
    const dray::ConstDeviceArray<Data> &samples,
    const dray::Vec<double, 3> &x) const
{
  const dray::Vec<double, 3> x_rel = x - this->m_origin;

  // project it
  dray::Vec<double, 2> coord = PlaneProject<double>(m_delta_a, m_delta_b).project(x_rel);
  for (int d : {0, 1})
    if (coord[d] >= 1.0f)
      coord[d] = 1.0f - dray::epsilon<Data>();

  // search for leaf and make coord relative to the leaf
  int node = 0;
  int level = 0;
  while (!this->m_quads[node].is_leaf())
  {
    coord *= 2;

    const int child_num = (int(coord[0]) << 0) | (int(coord[1]) << 1);
    node = this->m_quads[node].begin_children() + child_num;

    coord[0] -= trunc(coord[0]);
    coord[1] -= trunc(coord[1]);

    level++;
  }

  // interpolate
  double s00 = samples.get_item(4 * node + 0);
  double s01 = samples.get_item(4 * node + 1);
  double s10 = samples.get_item(4 * node + 2);
  double s11 = samples.get_item(4 * node + 3);

  double mid = s00 * (1-coord[0]) * (1-coord[1]) +
               s01 * (coord[0])   * (1-coord[1]) +
               s10 * (1-coord[0]) * (coord[1])   +
               s11 * (coord[0])   * (coord[1]);

  return mid;
}




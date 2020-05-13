// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include <dray/types.hpp>
#include <dray/Element/dof_access.hpp>
#include <dray/Element/element.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/elem_ops.hpp>

#include <iostream>
#include <array>
#include <algorithm>
#include <random>

TEST(dray_test_extern_eval, dray_test_extern_eval_d_edge)
{
  using dray::Float;
  using Vec1 = dray::Vec<Float, 1>;
  using Vec3 = dray::Vec<Float, 3>;
  using dray::Tensor;
  using dray::General;

  constexpr int max_order = 5;
  constexpr int max_npe3d = (max_order+1) * (max_order+1) * (max_order+1);

  // -->identity_map[]
  std::array<int, max_npe3d> identity_map;
  std::iota(identity_map.begin(), identity_map.end(), 0);

  // -->random_data[]
  std::array<Vec1, max_npe3d> random_data;
  const bool use_random = false;
  if (use_random)
  {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<Float> dist;
    std::generate(random_data.begin(), random_data.end(), std::bind(dist, gen));
  }
  else
  {
    std::iota(random_data.begin(), random_data.end(), 100);
  }

  // Compare evaluations for all orders.
  std::array<Float, max_npe3d+1> edge_samples;
  for (int p = 1; p <= max_order; ++p)
  {
    // -->edge_samples[]
    for (int i = 0; i <= p; ++i)
      edge_samples[i] = Float(double(i)/p);

    // -->rdp; (ReadDofPtr)
    dray::ReadDofPtr<Vec1> rdp;
    rdp.m_offset_ptr = identity_map.data();
    rdp.m_dof_ptr = random_data.data();

    // Create element.
    using ScalarElement = dray::Element<3, 1, Tensor, General>;
    ScalarElement elem = ScalarElement::create(0, rdp, p);

    // For each edge, compare evaluations.
    for (int e = 0; e < 12; ++e)
    {
      using namespace dray::hex_props;
      Vec3 sample3 = {{(Float) hex_eoffset0(e), (Float) hex_eoffset1(e), (Float) hex_eoffset2(e)}};
      const unsigned char eaxis = hex_eaxis(e);

      for (int i = 0; i <= p; ++i)
      {
        const Vec1 sample1 = {{(Float) edge_samples[i]}};
        sample3[eaxis] = edge_samples[i];
        dray::Vec<Vec1, 3> unused_deriv3;
        dray::Vec<Vec1, 1> unused_deriv1;

        const Vec1 eval_by_elem = elem.eval_d(sample3, unused_deriv3);
        const Vec1 eval_by_edge = dray::eops::eval_d_edge(
            dray::ShapeHex(), dray::OrderPolicy<General>{p}, e, rdp, sample1, unused_deriv1);

        if (p == 1)
        {
          const Vec1 eval_by_edge_fast = dray::eops::eval_d_edge(
              dray::ShapeHex(), dray::OrderPolicy<1>(), e, rdp, sample1, unused_deriv1);
          EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_edge_fast[0]);
        }
        if (p == 2)
        {
          const Vec1 eval_by_edge_fast = dray::eops::eval_d_edge(
              dray::ShapeHex(), dray::OrderPolicy<2>(), e, rdp, sample1, unused_deriv1);
          EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_edge_fast[0]);
        }

        EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_edge[0]);
      }
    }
  }
}

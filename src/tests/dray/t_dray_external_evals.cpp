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
  using Vec2 = dray::Vec<Float, 2>;
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
        dray::Vec<Vec1, 3> deriv3;
        dray::Vec<Vec1, 1> deriv1;

        const Vec1 eval_by_elem = elem.eval_d(sample3, deriv3);
        const Vec1 eval_by_edge = dray::eops::eval_d_edge(
            dray::ShapeHex(), dray::OrderPolicy<General>{p}, e, rdp, sample1, deriv1);
        EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_edge[0]);
        EXPECT_FLOAT_EQ(deriv3[eaxis][0], deriv1[0][0]);

        if (p == 1)
        {
          const Vec1 eval_by_edge_fast = dray::eops::eval_d_edge(
              dray::ShapeHex(), dray::OrderPolicy<1>(), e, rdp, sample1, deriv1);
          EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_edge_fast[0]);
          EXPECT_FLOAT_EQ(deriv3[eaxis][0], deriv1[0][0]);
        }
        if (p == 2)
        {
          const Vec1 eval_by_edge_fast = dray::eops::eval_d_edge(
              dray::ShapeHex(), dray::OrderPolicy<2>(), e, rdp, sample1, deriv1);
          EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_edge_fast[0]);
          EXPECT_FLOAT_EQ(deriv3[eaxis][0], deriv1[0][0]);
        }
      }
    }

    //STUB
    if (p != 1 && p != 2)
    {
      std::cout << "(p==" << p << ") Warning: General order face eval not tested yet\n";
    }

    // For each face, compare evaluations.
    for (int f = 0; f < 6; ++f)
    {
      using namespace dray::hex_props;
      Vec3 sample3 = {{(Float) hex_foffset0(f), (Float) hex_foffset1(f), (Float) hex_foffset2(f)}};
      const unsigned char faxisU = hex_faxisU(f);
      const unsigned char faxisV = hex_faxisV(f);

      for (int j = 0; j <= p; ++j)
        for (int i = 0; i <= p; ++i)
        {
          const Vec2 sample2 = {{(Float) edge_samples[i], (Float) edge_samples[j]}};
          sample3[faxisU] = edge_samples[i];
          sample3[faxisV] = edge_samples[j];
          dray::Vec<Vec1, 3> deriv3;
          dray::Vec<Vec1, 2> deriv2;

          const Vec1 eval_by_elem = elem.eval_d(sample3, deriv3);
          //TODO
          /// const Vec1 eval_by_face = dray::eops::eval_d_face(
          ///     dray::ShapeHex(), dray::OrderPolicy<General>{p}, f, rdp, sample2, deriv2);
          /// EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_face[0]);
          /// EXPECT_FLOAT_EQ(deriv3[faxisU][0], deriv2[0][0]);
          /// EXPECT_FLOAT_EQ(deriv3[faxisV][0], deriv2[1][0]);

          if (p == 1)
          {
            const Vec1 eval_by_face_fast = dray::eops::eval_d_face(
                dray::ShapeHex(), dray::OrderPolicy<1>(), f, rdp, sample2, deriv2);
            EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_face_fast[0]);
            EXPECT_FLOAT_EQ(deriv3[faxisU][0], deriv2[0][0]);
            EXPECT_FLOAT_EQ(deriv3[faxisV][0], deriv2[1][0]);
          }
          if (p == 2)
          {
            const Vec1 eval_by_face_fast = dray::eops::eval_d_face(
                dray::ShapeHex(), dray::OrderPolicy<2>(), f, rdp, sample2, deriv2);
            EXPECT_FLOAT_EQ(eval_by_elem[0], eval_by_face_fast[0]);
            EXPECT_FLOAT_EQ(deriv3[faxisU][0], deriv2[0][0]);
            EXPECT_FLOAT_EQ(deriv3[faxisV][0], deriv2[1][0]);
          }
        }
    }
  }
}

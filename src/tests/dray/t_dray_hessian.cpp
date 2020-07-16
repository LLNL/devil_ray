// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include <dray/types.hpp>
#include <dray/synthetic/affine_radial.hpp>

#include <dray/Element/element.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/elem_ops.hpp>

#include <dray/GridFunction/field.hpp>

#include <iostream>

TEST(dray_hessian, dray_zero_hessian)
{
  using dray::Float;

  const dray::Vec<int, 3> extents = {{4, 4, 4}};
  const dray::Vec<Float, 3> origin = {{0.0f, 0.0f, 0.0f}};
  const dray::Vec<Float, 3> radius = {{1.0f, 1.0f, 1.0f}};

  // Collection of hexs with no fields.
  dray::Collection collxn =
      dray::SynthesizeAffineRadial(extents, origin, radius)
      .synthesize();

  using ScalarFElemT = dray::Element<3, 1, dray::Tensor, dray::General>;

  // Add a constant field.
  const dray::Vec<Float, 1> constant = {{5.0f}};
  const std::string name = "VeryConstant";
  for (dray::DataSet &ds : collxn.domains())
    ds.add_field(std::make_shared<dray::Field<ScalarFElemT>>(
          dray::Field<ScalarFElemT>::uniform_field(ds.topology()->cells(), constant, name)));

}


TEST(dray_hessian, dray_constant_hessian)
{
  using dray::Float;

  const dray::Vec<int, 3> extents = {{4, 4, 4}};
  const dray::Vec<Float, 3> origin = {{0.0f, 0.0f, 0.0f}};
  const dray::Vec<Float, 3> radius = {{1.0f, 1.0f, 1.0f}};
  const dray::Vec<Float, 3> range_radius = {{1.0f, 1.0f, -1.0f}};
  const dray::Vec<Float, 3> range_radius_aux = {{1.0f, 1.0f, -1.0f}};

  dray::Collection collxn =
      dray::SynthesizeAffineRadial(extents, origin, radius)
      .equip("perfection", range_radius)
      .equip("aux", range_radius_aux)
      .synthesize();

}


TEST(dray_hessian, dray_grad_mag_grad)
{

}

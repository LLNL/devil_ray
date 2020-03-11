// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/templating/elem_attr.hpp>

#include <iostream>

TEST (dray_elem_attr, dray_elem_attr)
{
  dray::DefaultElemAttr default_elem_attr;

  using myElemAttr = dray::SetDimT<dray::DefaultElemAttr, dray::FixedDim<3>>;

  std::cout << "fixed_dim " << myElemAttr::fixed_dim << "\n";
  std::cout << "fixed_ncomp " << myElemAttr::fixed_ncomp << "\n";
  std::cout << "fixed_order " << myElemAttr::fixed_order << "\n";
  std::cout << "fixed_geom " << myElemAttr::fixed_geom << "\n";
  std::cout << "New dimension is " << myElemAttr::get_fixed_dim() << "\n";
}

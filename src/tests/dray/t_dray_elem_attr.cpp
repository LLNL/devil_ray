// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/templating/elem_attr.hpp>

#include <iostream>

template <class ElemAttrT>
void t_func(const ElemAttrT &elem_attr)
{
  std::cout << "dim.is_fixed " << elem_attr.dim.is_fixed << "\n";
  std::cout << "ncomp.is_fixed " << elem_attr.ncomp.is_fixed << "\n";
  std::cout << "order.is_fixed " << elem_attr.order.is_fixed << "\n";
  std::cout << "geom.is_fixed " << elem_attr.geom.is_fixed << "\n";
  std::cout << "New dimension is " << elem_attr.dim.m << "\n";

}

TEST (dray_elem_attr, dray_elem_attr)
{
  using MyElemAttr = dray::SetDimT<dray::DefaultElemAttr, dray::FixedDim<3>>;
  MyElemAttr elem_attr;
  elem_attr.ncomp.m = 1;
  elem_attr.order.m = 5;
  elem_attr.geom.m = dray::Hex;

  t_func(elem_attr);
}

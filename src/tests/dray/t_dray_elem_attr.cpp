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

struct MyOpFunctor
{
    template <typename NewT0, typename NewT1>  // Multiple template parameters ok, no state
    static void x(void *data, const NewT0 &a0, const NewT1 &a1)
    {
      std::cout << "a0 fixed: ";
      dray::dbg::print_fixed(std::cout, a0);
      std::cout << "  a0 name == " << a0 << "\n";

      std::cout << "a1 fixed: ";
      dray::dbg::print_fixed(std::cout, a1);
      std::cout << "  a1 name == " << a1 << "\n";

      std::cout << "a0.dim.is_fixed: " << a0.dim.is_fixed << ", a0.dim.m: " << a0.dim.m << "\n";
      std::cout << "a1.order.is_fixed: " << a1.order.is_fixed << ", a1.order.m: " << a1.order.m << "\n";
        // cast *data appropriately.
        // use the new constexpr wrapper types NewT0 and NewT1.
    }
};

struct MyOp
{
  void execute()
  {
    using namespace dray::dispatch;
    /// dray::WrapDim a0{3};
    /// dray::WrapOrder a1{2};
    dray::DefaultElemAttr a0, a1;
    a0.dim.m = 3;
    a1.order.m = 2;
    FixDim0< FixOrder1< MyOpFunctor >>::x(nullptr, a0, a1);
  }
};

TEST (dray_elem_attr, dray_elem_attr)
{
  using MyElemAttr = dray::SetDimT<dray::DefaultElemAttr, dray::FixedDim<3>>;
  MyElemAttr elem_attr;
  elem_attr.ncomp.m = 1;
  elem_attr.order.m = 5;
  elem_attr.geom.m = dray::Hex;

  t_func(elem_attr);

  MyOp().execute();
}

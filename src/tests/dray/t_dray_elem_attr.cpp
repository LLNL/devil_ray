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
    template <typename NewT1, typename NewT2>  // Multiple template parameters ok, no state
    static void x(void *data, const NewT1 &a1, const NewT2 &a2)
    {
      std::cout << "a1.is_fixed: " << a1.is_fixed << ", a1.m: " << a1.m << "\n";
      std::cout << "a2.is_fixed: " << a2.is_fixed << ", a2.m: " << a2.m << "\n";
        // cast *data appropriately.
        // use the new constexpr wrapper types NewT1 and NewT2.
    }
};

struct MyOp
{
  void execute()
  {
    using namespace dray::dispatch;
    dray::WrapDim a1{3};
    dray::WrapOrder a2{4};
    FixDim1< FixOrder2< MyOpFunctor >>::x(nullptr, a1, a2);
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

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ORDER_HPP
#define DRAY_ORDER_HPP

namespace dray
{

enum Order
{
  General = -1,
  Constant = 0,
  Linear = 1,
  Quadratic = 2,
  Cubic = 3,
};

} // namespace dray
#endif // DRAY_ORDER_HPP

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/fixed_point.hpp>
#include <ostream>

namespace dray
{
  std::ostream & operator<<(std::ostream &os, const FixedPoint &arg)
  {
    return os << float64(arg);
  }
}

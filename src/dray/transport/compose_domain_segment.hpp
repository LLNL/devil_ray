// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>

namespace dray
{
  Array<Float> compose_domain_segment(
      const Vec<Float, 3> &source,
      const Array<Vec<Float, 3>> &prefix_loc,
      const Array<Float> &prefix_avg_sigt,
      const Array<Vec<Float, 3>> &segment_exit_loc,
      const Array<Float> &segment_partial);
}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_PARTIALS_HPP
#define DRAY_UNIFORM_PARTIALS_HPP

/// #include <dray/data_model/data_set.hpp>
/// #include <dray/data_model/collection.hpp>
/// #include <dray/data_model/elem_utils.hpp>

#include <dray/uniform_topology.hpp>
#include <dray/data_model/low_order_field.hpp>


namespace dray
{
  // integrate absorption * length
  //   from source (or domain bdry) to world points
  //   (assume world points are in the given domain)
  std::pair<Array<Float>,
            Array<Vec<Float, 3>>>
  uniform_partials(
      const UniformTopology *mesh,
      const LowOrderField *absorption,
      const Vec<Float, 3> &source,
      const Array<Vec<Float, 3>> &world_points);

} // namespace dray



#endif//DRAY_UNIFORM_PARTIALS_HPP

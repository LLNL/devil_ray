// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_INTERPOLATION_SURFACE_HPP
#define DRAY_INTERPOLATION_SURFACE_HPP

#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

  // Abstract class interface for Interpolation Surfaces
  using SampleT = Float;
  class InterpolationSurface
  {
    public:
      /** Construct a surface over some number of planes. */
      InterpolationSurface(int32 num_planes);

      /** [in] plane_ids  [in] plane_vecs  [in] values */
      virtual void add_samples(
          Array<int32> plane_ids,
          Array<Vec<Float, 2>> plane_vecs,
          Array<SampleT> values) = 0;

      /** [in] plane_ids  [in] plane_vecs  [ret] values */
      virtual Array<SampleT> interpolate(
          Array<int32> plane_ids,
          Array<Vec<Float, 2>> plane_vecs) = 0;

      /** [out] plane_ids  [out] plane_vecs */
      virtual void suggested_samples(
          Float error,
          Array<int32> &plane_ids,
          Array<Vec<Float, 2>> &plane_vecs) = 0;
  };

}//namespace dray


#endif//DRAY_INTERPOLATION_SURFACE_HPP

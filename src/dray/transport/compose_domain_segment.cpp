// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/transport/compose_domain_segment.hpp>

#include <dray/policies.hpp>
#include <dray/exports.hpp>
#include <dray/device_array.hpp>
#include <dray/array_utils.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{
  Array<Float> compose_domain_segment(
      const Vec<Float, 3> &source_,
      const Array<Vec<Float, 3>> &prefix_loc,
      const Array<Float> &prefix_avg_sigt,
      const Array<Vec<Float, 3>> &segment_exit_loc,
      const Array<Float> &segment_partial)
  {
    const int32 size = segment_partial.size();

    Array<Float> composite_sigt = array_zero<Float>(size);

    ConstDeviceArray<Vec<Float, 3>> d_prefix_loc(prefix_loc);
    ConstDeviceArray<Vec<Float, 3>> d_segment_exit_loc(segment_exit_loc);
    ConstDeviceArray<Float> d_prefix_avg_sigt(prefix_avg_sigt);
    ConstDeviceArray<Float> d_segment_partial(segment_partial);
    NonConstDeviceArray<Float> d_composite_sigt(composite_sigt);

    const Vec<Float, 3> source = source_;  // avoid lambda issues

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      const Float prefix_avg_sigt = d_prefix_avg_sigt.get_item(i);
      const Float segment_partial = d_segment_partial.get_item(i);
      const Vec<Float, 3> entry_loc = d_prefix_loc.get_item(i);
      const Vec<Float, 3> exit_loc = d_segment_exit_loc.get_item(i);

      const Float dist_to_entry = (entry_loc - source).magnitude();
      const Float dist_to_exit = (exit_loc - source).magnitude();

      const Float composite_sigt =
        (prefix_avg_sigt * dist_to_entry + segment_partial)
            * rcp_safe(dist_to_exit);

      d_composite_sigt.get_item(i) = composite_sigt;
    });

    return composite_sigt;
  }
}

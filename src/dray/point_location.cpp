// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/point_location.hpp>

#include <dray/array_utils.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

#include <cstring>

namespace dray
{

PointLocator::PointLocator ()
{
}

PointLocator::PointLocator (BVH bvh)
{
  m_bvh = bvh;
}

PointLocator::~PointLocator ()
{
}

template <typename T>
PointLocator::Candidates
PointLocator::locate_candidates (const Array<Vec<T, 3>> points, const int max_candidates)
{
  const Array<int32> active_idx = array_counting (points.size (), 0, 1);
  return locate_candidates (points, active_idx, max_candidates);
}

template <typename T>
PointLocator::Candidates PointLocator::locate_candidates (const Array<Vec<T, 3>> points,
                                                          const Array<int32> active_idx,
                                                          const int max_candidates)
{
  DRAY_LOG_OPEN ("locate_candidates");
  Timer tot_timer;

  const int32 size_points = points.size ();
  const int32 size_active = active_idx.size ();

  Array<int32> candidates;
  Array<int32> aabb_ids;
  candidates.resize (size_active * max_candidates);
  aabb_ids.resize (size_active * max_candidates);

  array_memset (candidates, -1);
  int32 *candidates_ptr = candidates.get_device_ptr ();
  int32 *cand_aabb_id_ptr = aabb_ids.get_device_ptr ();

  const int32 *leaf_ptr = m_bvh.m_leaf_nodes.get_device_ptr_const ();
  const int32 *aabb_ids_ptr = m_bvh.m_aabb_ids.get_device_ptr_const ();
  const Vec<float32, 4> *inner_ptr = m_bvh.m_inner_nodes.get_device_ptr_const ();
  const Vec<T, 3> *points_ptr = points.get_device_ptr_const ();

  const int32 *active_idx_ptr = active_idx.get_device_ptr_const ();

  const int max_c = max_candidates;
  // std::cout<<"Point locator "<<size<<"\n";
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size_active), [=] DRAY_LAMBDA (int32 aii) {
    const int32 i = active_idx_ptr[aii];

    // - Use aii to index into candidates.
    // - Use i to index into points.

    int32 count = 0;
    int32 candidate_offset = aii * max_c;

    const Vec<T, 3> point = points_ptr[i];
    ;

    int32 current_node;
    int32 todo[64];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;
    while (current_node != barrier)
    {
      if (current_node > -1)
      {
        // inner node
        const Vec<float32, 4> first4 = const_get_vec4f (&inner_ptr[current_node + 0]);
        const Vec<float32, 4> second4 = const_get_vec4f (&inner_ptr[current_node + 1]);
        const Vec<float32, 4> third4 = const_get_vec4f (&inner_ptr[current_node + 2]);

        bool in_left = true;
        if (point[0] < first4[0]) in_left = false;
        if (point[1] < first4[1]) in_left = false;
        if (point[2] < first4[2]) in_left = false;

        if (point[0] > first4[3]) in_left = false;
        if (point[1] > second4[0]) in_left = false;
        if (point[2] > second4[1]) in_left = false;

        bool in_right = true;
        if (point[0] < second4[2]) in_right = false;
        if (point[1] < second4[3]) in_right = false;
        if (point[2] < third4[0]) in_right = false;

        if (point[0] > third4[1]) in_right = false;
        if (point[1] > third4[2]) in_right = false;
        if (point[2] > third4[3]) in_right = false;

        if (!in_left && !in_right)
        {
          // pop the stack and continue
          current_node = todo[stackptr];
          stackptr--;
        }
        else
        {
          Vec<float32, 4> children = const_get_vec4f (&inner_ptr[current_node + 3]);
          int32 l_child;
          constexpr int32 isize = sizeof (int32);
          // memcpy the int bits hidden in the floats
          memcpy (&l_child, &children[0], isize);
          int32 r_child;
          memcpy (&r_child, &children[1], isize);

          current_node = (in_left) ? l_child : r_child;

          if (in_left && in_right)
          {
            stackptr++;
            todo[stackptr] = r_child;
            // TODO: if we are in both children we could
            // go down the "closer" first by perhaps the distance
            // from the point to the center of the aabb
          }
        }
      }
      else
      {
        // leaf node
        // leafs are stored as negative numbers
        current_node = -current_node - 1; // swap the neg address
        candidates_ptr[candidate_offset + count] = leaf_ptr[current_node];
        cand_aabb_id_ptr[candidate_offset + count] = aabb_ids_ptr[current_node];
        count++;
        if (count == max_c)
        {
          /// printf("max candidates exceeded\n");    //TODO restore this debug
          break;
        }
        current_node = todo[stackptr];
        stackptr--;
      }
    } // while
  });
  DRAY_ERROR_CHECK();
  // std::cout<<"end Point locator\n";

  DRAY_LOG_ENTRY ("tot_time", tot_timer.elapsed ());
  DRAY_LOG_ENTRY ("num_points", size_points);
  DRAY_LOG_ENTRY ("num_active_points", size_active);
  DRAY_LOG_CLOSE ();
  PointLocator::Candidates res;
  res.m_candidates = candidates;
  res.m_aabb_ids = aabb_ids;
  return res;
}

// explicit instantiations
template PointLocator::Candidates
PointLocator::locate_candidates (const Array<Vec<float32, 3>> points, int32 max_candidates);

template PointLocator::Candidates
PointLocator::locate_candidates (const Array<Vec<float64, 3>> points, int32 max_candidates);

} // namespace dray

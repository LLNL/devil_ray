// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRIANGLE_MESH_HPP
#define DRAY_TRIANGLE_MESH_HPP

#include <dray/aabb.hpp>
#include <dray/array.hpp>
#include <dray/intersection_context.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/ray.hpp>

namespace dray
{

class TriangleMesh
{
  protected:
  Array<float32> m_coords;
  Array<int32> m_indices;
  BVH m_bvh;

  TriangleMesh ();

  public:
  TriangleMesh (Array<float32> &coords, Array<int32> &indices);
  ~TriangleMesh ();

  Array<RayHit> intersect (const Array<Ray> &rays);

  /**
   * @param[in] rays Rays that have already undergone the intersection test.
   *
   * \pre The Ray fields of m_dir, m_orig, m_dist, m_pixel_id, and m_hit_idx must be initialized.
   *
   * \retval intersection_ctx The intersection context for each ray.
   *   For any ray that does not intersect, the corresponding entry in m_is_valid is set to 0.
   */
  Array<IntersectionContext>
  get_intersection_context (const Array<Ray> &rays, const Array<RayHit> &hits);

  Array<float32> &get_coords ();
  Array<int32> &get_indices ();
  AABB<> get_bounds ();
};

} // namespace dray

#endif

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRIANGLE_MESH_HPP
#define DRAY_TRIANGLE_MESH_HPP

#include <dray/aabb.hpp>
#include <dray/array.hpp>
#include <dray/rendering/fragment.hpp>
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

  //Array<Fragment>
  //fragments(const Array<Ray> &rays, const Array<RayHit> &hits);

  Array<float32> &get_coords ();
  Array<int32> &get_indices ();
  AABB<> get_bounds ();
};

} // namespace dray

#endif

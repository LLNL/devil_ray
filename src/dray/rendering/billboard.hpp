// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BILLBOARD_HPP
#define DRAY_BILLBOARD_HPP

#include <dray/exports.hpp>
#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/rendering/framebuffer.hpp>

namespace dray
{

class Billboard
{
protected:
  Array<Vec<float32,3>> m_centers;
  Array<Vec<float32,2>> m_dims; // width and height of each billboard
  Array<Vec<float32,2>> m_tcoords;
  BVH m_bvh;
public:
  Billboard(const std::string text, const Vec<float32,3> pos);

  Array<RayHit> intersect (const Array<Ray> &rays);
  void shade(const Array<Ray> &rays, const Array<RayHit> &hits, Framebuffer &fb);
  AABB<3> bounds();

};


} // namespace dray
#endif

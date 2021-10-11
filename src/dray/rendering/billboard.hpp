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
#include <dray/rendering/camera.hpp>

namespace dray
{

class Billboard
{
protected:
  Vec<float32,3> m_up;
  Vec<float32,3> m_ray_differential_x;
  Vec<float32,3> m_ray_differential_y;
  Vec<float32,3> m_text_color;
  Array<Vec<float32,3>> m_centers;
  Array<Vec<float32,2>> m_dims; // width and height of each billboard
  Array<Vec<float32,2>> m_tcoords;
  Array<float32> m_texture;
  int32 m_texture_width;
  int32 m_texture_height;
  BVH m_bvh;
public:
  Billboard() = delete;
  Billboard(const std::vector<std::string> &texts,
            const std::vector<Vec<float32,3>> &positions,
            const std::vector<float32> &world_sizes);

  void camera(const Camera& camera);
  void text_color(const Vec<float32,3> &color);
  Array<RayHit> intersect (const Array<Ray> &rays);
  void shade(const Array<Ray> &rays, const Array<RayHit> &hits, Framebuffer &fb);
  AABB<3> bounds();

  friend struct DeviceBillboard;
};


} // namespace dray
#endif

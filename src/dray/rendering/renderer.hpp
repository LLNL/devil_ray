// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RENDERER_HPP
#define DRAY_RENDERER_HPP

#include <dray/rendering/camera.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/point_light.hpp>
#include <dray/rendering/traceable.hpp>

#include <memory>
#include <vector>

namespace dray
{

class Renderer
{
protected:
  std::vector<std::shared_ptr<Traceable>> m_traceables;
  std::vector<PointLight> m_lights;
public:
  void ray_max(Array<Ray> &rays, const Array<RayHit> &hits) const;
  void clear();
  void clear_lights();
  void add(std::shared_ptr<Traceable> traceable);
  void add_light(const PointLight &light);
  Framebuffer render(Camera &camera);
};


} // namespace dray
#endif

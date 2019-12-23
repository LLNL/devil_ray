// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/ray_tracing/renderer.hpp>

#include <memory>
#include <vector>

namespace dray
{
namespace ray_tracing
{

void Renderer::clear()
{
  m_traceables.clear();
}

void Renderer::add(std::shared_ptr<Traceable> &traceable)
{
  m_traceables.push_back(traceable);
}

Framebuffer Renderer::render(Camera &camera)
{
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::Framebuffer framebuffer (camera.get_width(), camera.get_height());
  framebuffer.clear ();

  const int32 size = m_traceables.size();

  for(int i = 0; i < size; ++i)
  {
    Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
    Array<Fragment> fragments = m_traceables[i]->fragments(hits);
  }

  Framebuffer fb;
  return fb;
}


}} // namespace dray::ray_tracing

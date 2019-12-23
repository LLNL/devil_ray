// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/ray_tracing/renderer.hpp>
#include <dray/policies.hpp>

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

void Renderer::clear_lights()
{
  m_lights.clear();
}

void Renderer::add_light(const PointLight &light)
{
  m_lights.push_back(light);
}

void Renderer::add(std::shared_ptr<Traceable> traceable)
{
  m_traceables.push_back(traceable);
}

Framebuffer Renderer::render(Camera &camera)
{
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::Framebuffer framebuffer (camera.get_width(), camera.get_height());
  framebuffer.clear ();

  Array<PointLight> lights;
  lights.resize(m_lights.size());
  PointLight* light_ptr = lights.get_host_ptr();
  for(int i = 0; i < m_lights.size(); ++i)
  {
    light_ptr[i] = m_lights[i];
  }

  const int32 size = m_traceables.size();

  for(int i = 0; i < size; ++i)
  {
    std::cout<<"A\n";
    Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
    std::cout<<"B "<<hits.size()<<"\n";
    Array<Fragment> fragments = m_traceables[i]->fragments(hits);
    std::cout<<"C\n";

    m_traceables[i]->shade(rays, hits, fragments, lights, framebuffer);
    // set ray max dist
  }

  return framebuffer;
}

void Renderer::ray_max(Array<Ray> &rays, const Array<RayHit> &hits) const
{
  const int32 size = rays.size();
  Ray *ray_ptr = rays.get_device_ptr();
  const RayHit *hit_ptr = hits.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const RayHit hit = hit_ptr[i];
    if(hit.m_hit_idx != -1)
    {
      ray_ptr[i].m_far = hit.m_dist;
    }

  });
}

}} // namespace dray::ray_tracing

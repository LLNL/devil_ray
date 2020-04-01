// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>

#include <memory>
#include <vector>

namespace dray
{

namespace detail
{
PointLight default_light(Camera &camera)
{
  Vec<float32,3> look_at = camera.get_look_at();
  Vec<float32,3> pos = camera.get_pos();
  Vec<float32,3> up = camera.get_up();
  up.normalize();
  Vec<float32,3> look = look_at - pos;
  float32 mag = look.magnitude();
  Vec<float32,3> right = cross (look, up);
  right.normalize();

  Vec<float32, 3> miner_up = cross (right, look);

  Vec<float32, 3> light_pos = pos + .3f * mag * miner_up;
  PointLight light;
  light.m_pos = light_pos;
  return light;
}

} // namespace detail

Renderer::Renderer()
  : m_use_lighting(true)
{
}

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

void Renderer::use_lighting(bool use_it)
{
  m_use_lighting = use_it;
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
  if(m_lights.size() > 0)
  {
    lights.resize(m_lights.size());
    PointLight* light_ptr = lights.get_host_ptr();
    for(int i = 0; i < m_lights.size(); ++i)
    {
      light_ptr[i] = m_lights[i];
    }
  }
  else
  {
    lights.resize(1);
    PointLight light = detail::default_light(camera);
    PointLight* light_ptr = lights.get_host_ptr();
    light_ptr[0] = light;
  }

  const int32 size = m_traceables.size();

  int32 volume_index = -1;
  for(int i = 0; i < size; ++i)
  {
    bool is_volume = m_traceables[i]->is_volume();
    if(is_volume && volume_index == -1)
    {
      volume_index = i;
    }
    else if(is_volume)
    {
      DRAY_ERROR("Only a single volume is supported");
    }
  }

  for(int i = 0; i < size; ++i)
  {
    if(i == volume_index)
    {
      continue;
    }
    Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
    Array<Fragment> fragments = m_traceables[i]->fragments(hits);
    if(m_use_lighting)
    {
      m_traceables[i]->shade(rays, hits, fragments, lights, framebuffer);
    }
    else
    {
      m_traceables[i]->shade(rays, hits, fragments, framebuffer);
    }

    ray_max(rays, hits);
  }

  if(volume_index > -1)
  {
    Volume* volume = dynamic_cast<Volume*>(m_traceables[volume_index].get());
    volume->integrate(rays, framebuffer, lights);
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
  DRAY_ERROR_CHECK();
}

} // namespace dray

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>

#include <memory>
#include <vector>

namespace dray
{

void ScalarRenderer::set(std::shared_ptr<Traceable> traceable)
{
  bool is_volume = traceable->is_volume();
  if(is_volume)
  {
    DRAY_ERROR("Scalar renderer does not support volumes.");
  }

  m_traceable = traceable;
}

Framebuffer ScalarRenderer::render(Camera &camera)
{
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::Framebuffer framebuffer (camera.get_width(), camera.get_height());
  framebuffer.clear ();

  Array<RayHit> hits = m_traceable->nearest_hit(rays);
  Array<Fragment> fragments = m_traceable->fragments(hits);
  //m_traceables[i]->shade(rays, hits, fragments, lights, framebuffer);
  //ray_max(rays, hits);

  return framebuffer;
}

void ScalarRenderer::ray_max(Array<Ray> &rays, const Array<RayHit> &hits) const
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

} // namespace dray

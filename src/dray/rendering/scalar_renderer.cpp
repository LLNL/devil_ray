// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/device_scalar_buffer.hpp>
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

ScalarBuffer
ScalarRenderer::render(Camera &camera)
{
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  ScalarBuffer scalar_buffer(camera.get_width(), camera.get_height());
  scalar_buffer.clear();

  Array<RayHit> hits = m_traceable->nearest_hit(rays);
  Array<Fragment> fragments = m_traceable->fragments(hits);

  // extract the scalars
  DeviceScalarBuffer d_buffer(scalar_buffer);

  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const Fragment *frag_ptr = fragments.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];
    const Ray &ray = ray_ptr[ii];

    if (hit.m_hit_idx > -1)
    {
      const int32 pid = ray.m_pixel_id;
      d_buffer.m_scalars[pid] = frag.m_scalar;
      d_buffer.m_depths[pid] = hit.m_dist;
    }

  });

  return scalar_buffer;
}

} // namespace dray

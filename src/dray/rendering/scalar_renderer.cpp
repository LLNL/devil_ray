// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_renderer.hpp>

#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/policies.hpp>

#include <memory>
#include <vector>

namespace dray
{

namespace
{

template<typename FloatType>
void init_buffer(Array<FloatType> &scalars, const FloatType clear_value)
{
  const int32 size = scalars.size();
  FloatType *scalar_ptr = scalars.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii) {
    scalar_ptr[ii] = clear_value;
  });
  DRAY_ERROR_CHECK();
}

} // namespace

void ScalarRenderer::set(std::shared_ptr<Traceable> traceable)
{
  bool is_volume = traceable->is_volume();
  if(is_volume)
  {
    DRAY_ERROR("Scalar renderer does not support volumes.");
  }

  m_traceable = traceable;
}

void ScalarRenderer::field_names(const std::vector<std::string> &field_names)
{
  m_field_names = field_names;
}

ScalarBuffer
ScalarRenderer::render(Camera &camera)
{
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  Array<RayHit> hits = m_traceable->nearest_hit(rays);

  ScalarBuffer scalar_buffer;
  scalar_buffer.m_width = camera.get_width();
  scalar_buffer.m_height = camera.get_height();
  scalar_buffer.m_clear_value = camera.get_height();

  const int32 buffer_size = camera.get_width() * camera.get_height();
  const int32 field_size = m_field_names.size();

  scalar_buffer.m_depths.resize(buffer_size);
  init_buffer(scalar_buffer.m_depths, nan<float32>());

  float32 *depth_ptr = scalar_buffer.m_depths.get_device_ptr();

  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const RayHit *hit_ptr = hits.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Ray &ray = ray_ptr[ii];
    if (hit.m_hit_idx > -1)
    {
      const int32 pid = ray.m_pixel_id;
      depth_ptr[pid] = hit.m_dist;
    }
  });

  for(int32 i = 0; i < field_size; ++i)
  {
    std::string field = m_field_names[i];
    m_traceable->field(field);
    Array<Fragment> fragments = m_traceable->fragments(hits);
    Array<float32> buffer;
    buffer.resize(buffer_size);
    init_buffer(buffer, nan<float32>());
    float32 *buffer_ptr = buffer.get_device_ptr();

    const Fragment *frag_ptr = fragments.get_device_ptr_const ();
    RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
    {
      const RayHit &hit = hit_ptr[ii];
      const Fragment &frag = frag_ptr[ii];
      const Ray &ray = ray_ptr[ii];

      if (hit.m_hit_idx > -1)
      {
        const int32 pid = ray.m_pixel_id;
        buffer_ptr[pid] = frag.m_scalar;
      }

    });

    scalar_buffer.m_scalars.push_back(buffer);
    scalar_buffer.m_names.push_back(field);
  }

  return scalar_buffer;
}

} // namespace dray

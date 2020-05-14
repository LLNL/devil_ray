// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_renderer.hpp>

#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/policies.hpp>

#include <memory>
#include <vector>

#include <apcomp/scalar_compositor.hpp>

namespace dray
{

namespace
{

ScalarBuffer convert(apcomp::ScalarImage &image, std::vector<std::string> &names)
{
  ScalarBuffer result;
  result.m_names = names;
  const int num_fields = names.size();

  const int dx  = image.m_bounds.m_max_x - image.m_bounds.m_min_x + 1;
  const int dy  = image.m_bounds.m_max_y - image.m_bounds.m_min_y + 1;
  const int size = dx * dy;

  result.m_width = dx;
  result.m_height = dy;

  std::vector<float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    Array<float> array;
    array.resize(size);
    result.m_scalars.push_back(array);
    float* buffer = result.m_scalars[i].get_host_ptr();
    buffers.push_back(buffer);
  }

  const unsigned char *loads = &image.m_payloads[0];
  const size_t payload_size = image.m_payload_bytes;

#ifdef DRAY_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(float);
      memcpy(&buffers[i][x], loads + offset, sizeof(float));
    }
  }

  result.m_depths.resize(size);
  float* dbuffer = result.m_depths.get_host_ptr();
  memcpy(dbuffer, &image.m_depths[0], sizeof(float) * size);

  return result;
}

apcomp::ScalarImage * convert(ScalarBuffer &result)
{
  const int num_fields = result.m_scalars.size();
  const int payload_size = num_fields * sizeof(float);
  apcomp::Bounds bounds;
  bounds.m_min_x = 1;
  bounds.m_min_y = 1;
  bounds.m_max_x = result.m_width;
  bounds.m_max_y = result.m_height;
  const size_t size = result.m_width * result.m_height;

  apcomp::ScalarImage *image = new apcomp::ScalarImage(bounds, payload_size);
  unsigned char *loads = &image->m_payloads[0];

  const float* dbuffer = result.m_depths.get_host_ptr();
  memcpy(&image->m_depths[0], dbuffer, sizeof(float) * size);
  // copy scalars into payload
  std::vector<float*> buffers;
  for(int i = 0; i < num_fields; ++i)
  {
    float* buffer = result.m_scalars[i].get_host_ptr();
    buffers.push_back(buffer);
  }
#ifdef DRAY_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(size_t x = 0; x < size; ++x)
  {
    for(int i = 0; i < num_fields; ++i)
    {
      const size_t offset = x * payload_size + i * sizeof(float);
      memcpy(loads + offset, &buffers[i][x], sizeof(float));
    }
  }
  return image;
}


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


ScalarRenderer::ScalarRenderer()
  : m_traceable(nullptr)
{
}

ScalarRenderer::ScalarRenderer(std::shared_ptr<Traceable> traceable)
  : m_traceable(traceable)
{
}

void ScalarRenderer::set(std::shared_ptr<Traceable> traceable)
{
  m_traceable = traceable;
}

void ScalarRenderer::field_names(const std::vector<std::string> &field_names)
{
  m_field_names = field_names;
}

ScalarBuffer
ScalarRenderer::render(Camera &camera)
{
  if(m_traceable == nullptr)
  {
    DRAY_ERROR("ScalarRenderer: traceable never set");
  }

  Array<Ray> rays;
  camera.create_rays (rays);

  Array<RayHit> hits = m_traceable->nearest_hit(rays);

  std::vector<ScalarBuffer> buffers;
  const int domains = m_traceable->num_domains();

  for(int d = 0; d < domains; ++d)
  {
    m_traceable->active_domain(d);

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

    buffers.push_back(scalar_buffer);
  }

  // basic sanity checking
  int min_p = std::numeric_limits<int>::max();
  int max_p = std::numeric_limits<int>::min();

  apcomp::PayloadCompositor compositor;

  for(auto buffer : buffers)
  {
    apcomp::ScalarImage *pimage = convert(buffer);

    min_p = std::min(min_p, pimage->m_payload_bytes);
    max_p = std::max(max_p, pimage->m_payload_bytes);

    compositor.AddImage(*pimage);
    delete pimage;
  }

  if(min_p != max_p)
  {
    DRAY_ERROR("VERY BAD "<<min_p<<" "<<max_p<<" panic and contact someone.");
  }

  // only valid on rank 0
  apcomp::ScalarImage final_image = compositor.Composite();
  ScalarBuffer final_result;
  if(dray::mpi_rank() == 0)
  {
    final_result = convert(final_image, m_field_names);
  }

  return final_result;
}

} // namespace dray

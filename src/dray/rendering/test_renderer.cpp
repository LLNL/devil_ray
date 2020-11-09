// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/test_renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/rendering/annotator.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/dray.hpp>
#include <dray/array_utils.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>
#include <dray/device_color_map.hpp>

#include <memory>
#include <vector>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

static float32 ray_eps = 1e-5;

namespace detail
{

void multiply(Array<Vec<float32,3>> &input, Array<Vec<float32,4>> &factor)
{
  const Vec<float32,4> *factor_ptr = factor.get_device_ptr_const();
  Vec<float32,3> *input_ptr = input.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, input.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Vec<float32,4> f = factor_ptr[ii];
    Vec<float32,3> in = input_ptr[ii];
    in[0] *= f[0];
    in[1] *= f[1];
    in[2] *= f[2];
    input_ptr[ii] = in;
  });
  DRAY_ERROR_CHECK();
}

struct DeviceSamples
{
  Vec<float32,4> *colors;
  Vec<float32,3> *normals;
  float32 *distances;
  int32 *hit_flags;

  DeviceSamples() = delete;
  DeviceSamples(Samples &samples)
  {
    colors = samples.m_colors.get_device_ptr();
    normals = samples.m_normals.get_device_ptr();
    distances = samples.m_distances.get_device_ptr();
    hit_flags = samples.m_hit_flags.get_device_ptr();
  }
};


void compact_hits(Array<Ray> &rays, Samples &samples, Array<Vec<float32,3>> &intensities)
{
  Array<int32> compact_idxs = index_flags(samples.m_hit_flags);

  rays = gather(rays, compact_idxs);
  samples.m_colors = gather(samples.m_colors, compact_idxs);
  samples.m_normals = gather(samples.m_normals, compact_idxs);
  samples.m_distances = gather(samples.m_distances, compact_idxs);
  samples.m_hit_flags = gather(samples.m_hit_flags, compact_idxs);
  intensities = gather(intensities, compact_idxs);
}

void update_hits( Array<RayHit> &hits,
                  Array<int32> &hit_flags)
{
  DRAY_LOG_OPEN("update_hits");

  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  int32 *hit_flag_ptr = hit_flags.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];

    if (hit.m_hit_idx > -1)
    {
      hit_flag_ptr[ii] = 1;
    }

  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

void update_samples( Array<RayHit> &hits,
                     Array<Fragment> &fragments,
                     ColorMap &color_map,
                     Samples &samples)
{
  DRAY_LOG_OPEN("update_samples");

  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  const Fragment *frag_ptr = fragments.get_device_ptr_const ();

  DeviceSamples d_samples(samples);
  DeviceColorMap d_color_map (color_map);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];

    // could check the current distance
    // but we adjust the ray max do this shouldn't need to
    if (hit.m_hit_idx > -1)
    {
      const Float sample_val = frag.m_scalar;
      Vec4f sample_color = d_color_map.color (sample_val);

      d_samples.normals[ii] = frag.m_normal;
      d_samples.colors[ii]  = sample_color;
      d_samples.distances[ii] = hit.m_dist;
      d_samples.hit_flags[ii] = 1;
    }

  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

PointLight test_default_light(Camera &camera)
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
  miner_up.normalize();

  Vec<float32, 3> light_pos = pos + .1f * mag * miner_up;
  PointLight light;
  light.m_pos = light_pos;
  return light;
}

} // namespace detail

void Samples::resize(int32 size)
{
  m_colors.resize(size);
  m_normals.resize(size);
  m_distances.resize(size);
  m_hit_flags.resize(size);
  array_memset_zero (m_hit_flags);
}

TestRenderer::TestRenderer()
  : m_volume(nullptr),
    m_use_lighting(true),
    m_screen_annotations(true)
{
}

void TestRenderer::clear()
{
  m_traceables.clear();
}

void TestRenderer::screen_annotations(bool on)
{
  m_screen_annotations = on;
}

void TestRenderer::clear_lights()
{
  m_lights.clear();
}

void TestRenderer::add_light(const PointLight &light)
{
  m_lights.push_back(light);
}

void TestRenderer::use_lighting(bool use_it)
{
  m_use_lighting = use_it;
}

void TestRenderer::add(std::shared_ptr<Traceable> traceable)
{
  m_traceables.push_back(traceable);
}


void TestRenderer::volume(std::shared_ptr<Volume> volume)
{
  m_volume = volume;
}

Array<int32> TestRenderer::any_hit(Array<Ray> &rays)
{
  const int32 size = m_traceables.size();

  Array<int32> hit_flags;
  hit_flags.resize(rays.size());
  array_memset_zero(hit_flags);

  for(int i = 0; i < size; ++i)
  {
    const int domains = m_traceables[i]->num_domains();
    for(int d = 0; d < domains; ++d)
    {
      m_traceables[i]->active_domain(d);
      // TODO: actually make and any hit method
      Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
      detail::update_hits(hits, hit_flags);
      ray_max(rays, hits);
    }
  }

  return hit_flags;
}

Samples TestRenderer::nearest_hits(Array<Ray> &rays)
{
  const int32 size = m_traceables.size();

  Samples samples;
  samples.resize(rays.size());

  for(int i = 0; i < size; ++i)
  {
    const int domains = m_traceables[i]->num_domains();
    for(int d = 0; d < domains; ++d)
    {
      m_traceables[i]->active_domain(d);
      Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
      Array<Fragment> fragments = m_traceables[i]->fragments(hits);
      detail::update_samples(hits, fragments, m_traceables[i]->color_map(), samples);
      ray_max(rays, hits);
    }
  }

  return samples;
}

Framebuffer TestRenderer::render(Camera &camera)
{
  DRAY_LOG_OPEN("render");
  Array<Ray> rays;
  camera.create_rays (rays);

  std::vector<std::string> field_names;
  std::vector<ColorMap> color_maps;

  AABB<3> scene_bounds;
  for(int i = 0; i < m_traceables.size(); ++i)
  {
    scene_bounds.include(m_traceables[i]->collection().bounds());
  }

  ray_eps = scene_bounds.max_length() * 1e-6;

  Framebuffer framebuffer (camera.get_width(), camera.get_height());
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
    PointLight light = detail::test_default_light(camera);
    PointLight* light_ptr = lights.get_host_ptr();
    light_ptr[0] = light;
  }


  Array<Vec<float32,3>> attenuation;
  attenuation.resize(rays.size());
  Vec<float32,3> white = {{1.f,1.f,1.f}};
  array_memset_vec(attenuation, white);

  Samples samples = nearest_hits(rays);
  // reduce to only the hits
  detail::compact_hits(rays, samples, attenuation);

  // cast shadow rays
  Array<Vec<float32,3>> light_colors = direct_lighting(lights, rays, samples);

  // attenuate the light
  detail::multiply(attenuation, samples.m_colors);

  // shade and bounce;

  // get stuff for annotations
  for(int i = 0; i < m_traceables.size(); ++i)
  {
    field_names.push_back(m_traceables[i]->field());
    color_maps.push_back(m_traceables[i]->color_map());
  }

  if(m_screen_annotations && dray::mpi_rank() == 0)
  {
    Annotator annot;
    annot.screen_annotations(framebuffer, field_names, color_maps);
  }
  DRAY_LOG_CLOSE();

  return framebuffer;
}
void TestRenderer::bounce(Array<Ray> &rays, Samples &samples)
{

}

Array<Ray>
TestRenderer::create_shadow_rays(Array<Ray> &rays,
                                 Array<float32> &distances,
                                 const Vec<float32,3> point)
{
  Array<Ray> shadow_rays;
  shadow_rays.resize(rays.size());
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const float32 *distances_ptr = distances.get_device_ptr_const ();
  Ray *shadow_ptr = shadow_rays.get_device_ptr();

  const float32 eps = ray_eps;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray &ray = ray_ptr[ii];
    const float32 distance = distances_ptr[ii];
    Ray shadow_ray;
    Vec<float32,3> hit_point = ray.m_orig + ray.m_dir * distance;
    // back it away a bit
    hit_point += eps * (-ray.m_dir);

    shadow_ray.m_orig = hit_point;
    shadow_ray.m_near = 0.f;

    Vec<float32, 3> light_dir = point - hit_point;
    shadow_ray.m_far = light_dir.magnitude();
    light_dir.normalize();
    shadow_ray.m_dir = light_dir;

    shadow_ray.m_pixel_id = ray.m_pixel_id;
    shadow_ptr[ii] = shadow_ray;
  });

  return shadow_rays;
}


void
TestRenderer::shade_lights(const Vec<float32,3> light_color,
                           Array<Ray> &rays,
                           Array<Ray> &shadow_rays,
                           Array<Vec<float32,3>> &normals,
                           Array<int32> &hit_flags,
                           Array<Vec<float32,3>> &colors)
{
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const Ray *shadow_ptr = shadow_rays.get_device_ptr_const();
  const int32 *hit_flag_ptr = hit_flags.get_device_ptr_const();
  const Vec<float32,3> *normal_ptr = normals.get_device_ptr_const();
  Vec<float32,3> *color_ptr = colors.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray &ray = ray_ptr[ii];
    const Ray &shadow_ray = shadow_ptr[ii];
    Vec<float32,3> normal = normal_ptr[ii];
    normal.normalize();

    Vec<float32,3> color = {{0.f, 0.f, 0.f}};

    if(hit_flag_ptr[ii] != 0)
    {
      color = light_color * max(0.f, dot(normal, shadow_ray.m_dir));
    }

    color_ptr[ii] += color;

  });
}

Array<Vec<float32,3>>
TestRenderer::direct_lighting(Array<PointLight> &lights,
                              Array<Ray> &rays,
                              Samples &samples)
{
  Array<Vec<float32,3>> contributions;
  contributions.resize(rays.size());
  Vec<float32,3> black = {{0.f, 0.f, 0.f}};
  array_memset_vec (contributions, black);

  for(int l = 0; l < lights.size(); ++l)
  {
    PointLight light = lights.get_value(l);
    Array<Ray> shadow_rays = create_shadow_rays(rays, samples.m_distances, light.m_pos);
    Array<int32> hit_flags = any_hit(shadow_rays);

    shade_lights(light.m_diff,
                 rays,
                 shadow_rays,
                 samples.m_normals,
                 hit_flags,
                 contributions);
  }

  // multiply the light contributions by the input color
  detail::multiply(contributions, samples.m_colors);
  return contributions;

}

} // namespace dray

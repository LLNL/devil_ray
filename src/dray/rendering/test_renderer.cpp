// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/test_renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/rendering/annotator.hpp>
#include <dray/rendering/low_order_intersectors.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/rendering/sampling.hpp>
#include <dray/rendering/disney_sampling.hpp>
#include <dray/rendering/device_env_map.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/dray.hpp>
#include <dray/array_utils.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>
#include <dray/device_color_map.hpp>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <memory>
#include <vector>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

#define RAY_DEBUGGING

//int debug_ray = 160448;

namespace dray
{

namespace detail
{

Array<int32> radiance_ray_indices(Array<RayData> &data)
{
  const int size = data.size();
  const RayData *data_ptr = data.get_device_ptr_const();

  Array<int32> flags;
  flags.resize(size);
  int32 *flags_ptr = flags.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    int32 flag = 1;
    if(data_ptr[ii].m_flags == RayFlags::TRANSMITTANCE)
    {
      flag = 0;
    }
    flags_ptr[ii] = flag;
  });
  DRAY_ERROR_CHECK();

  return index_flags(flags);
}

template<int32 T>
void multiply(Array<Vec<float32,3>> &input, Array<Vec<float32,T>> &factor)
{
  const Vec<float32,T> *factor_ptr = factor.get_device_ptr_const();
  Vec<float32,3> *input_ptr = input.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, input.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Vec<float32,T> f = factor_ptr[ii];
    Vec<float32,3> in = input_ptr[ii];
    in[0] *= f[0];
    in[1] *= f[1];
    in[2] *= f[2];
    input_ptr[ii] = in;
  });
  DRAY_ERROR_CHECK();
}

void multiply(Array<Vec<float32,3>> &input, Array<Sample> &samples)
{
  const Sample *sample_ptr = samples.get_device_ptr_const();
  Vec<float32,3> *input_ptr = input.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, input.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Vec<float32,4> f = sample_ptr[ii].m_color;
    Vec<float32,3> in = input_ptr[ii];
    in[0] *= f[0];
    in[1] *= f[1];
    in[2] *= f[2];
    input_ptr[ii] = in;
  });
  DRAY_ERROR_CHECK();
}

void multiply(Array<RayData> &data, Array<Sample> &samples)
{
  const Sample *sample_ptr = samples.get_device_ptr_const();
  RayData *data_ptr = data.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, data.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Vec<float32,4> f = sample_ptr[ii].m_color;
    RayData data = data_ptr[ii];
    data.m_throughput[0] *= f[0];
    data.m_throughput[1] *= f[1];
    data.m_throughput[2] *= f[2];
    data_ptr[ii] = data;
  });
  DRAY_ERROR_CHECK();
}
#warning "get rid of all the multiplys and just add this stuff into funcs"
void multiply(Array<Vec<float32,3>> &input, Array<RayData> &data)
{
  const RayData *data_ptr = data.get_device_ptr_const();
  Vec<float32,3> *input_ptr = input.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, input.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Vec<float32,3> f = data_ptr[ii].m_throughput;
    Vec<float32,3> in = input_ptr[ii];
    in[0] *= f[0];
    in[1] *= f[1];
    in[2] *= f[2];
    input_ptr[ii] = in;
  });
  DRAY_ERROR_CHECK();
}

Array<int32> extract_hit_flags(Array<Sample> &samples)
{
  Array<int32> flags;
  flags.resize(samples.size());
  int32 *output_ptr = flags.get_device_ptr();
  const Sample *sample_ptr = samples.get_device_ptr_const();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, samples.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    output_ptr[ii] = sample_ptr[ii].m_hit_flag;
  });

  DRAY_ERROR_CHECK();
  return flags;
}

void process_shadow_rays(Array<Ray> &rays,
                         Array<RayData> &ray_data,
                         Array<Sample> &samples,
                         Array<Vec<float32,4>> &color_buffer,
                         Array<Material> &materials,
                         const float32 scene_eps,
                         const int32 debug_ray)
{
  Ray *ray_ptr = rays.get_device_ptr();
  RayData *data_ptr = ray_data.get_device_ptr();
  const Material * mat_ptr = materials.get_device_ptr_const();
  const float32 eps = scene_eps;

  Sample *sample_ptr = samples.get_device_ptr();
  Vec<float32,4> *color_ptr = color_buffer.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
     bool debug = ray_ptr[ii].m_pixel_id == debug_ray;
     // TODO: this logic can be a lot cleaner

     RayData data = data_ptr[ii];
     if(data.m_flags == RayFlags::TRANSMITTANCE)
     {
        Sample sample = sample_ptr[ii];

        if(sample.m_hit_flag == 0)
        {
          Vec<float32,4> color;
          color[0] = data.m_throughput[0];
          color[1] = data.m_throughput[1];
          color[2] = data.m_throughput[2];
          color[3] = 1.f;
          if(debug)
          {
            std::cout<<"[Direct lighting] adding "<<color<<"\n";
          }
          color_ptr[ray_ptr[ii].m_pixel_id] += color;
        }
        else
        {
          // ok we hit something and we need to check if its transparent or not
          // TODO: future spec trans will part of the transfer function somehow
          Material mat = mat_ptr[sample.m_mat_id];
          bool keep = 1;

          if(mat.m_spec_trans > 0.f)
          {
            if(debug)
            {
              std::cout<<"[shadow processing] transparency\n";
            }

            Vec<float32,4> color = sample.m_color;
            Vec<float32,3> base_color = {{color[0],color[1],color[2]}};
            Ray ray = ray_ptr[ii];

            Vec<float32,3> normal = sample.m_normal;
            if(dot(ray.m_dir, normal) > 0)
            {
              normal = -normal;
            }

            Vec<float32,3> wcX, wcY;
            create_basis(normal,wcX,wcY);
            Matrix<float32,3,3> to_world, to_tangent;
            to_world.set_col(0,wcX);
            to_world.set_col(1,wcY);
            to_world.set_col(2,normal);
            to_tangent = to_world.transpose();

            Vec<float32,3> wo = to_tangent * (-ray.m_dir);
            Vec<float32,3> wi = -wo;
            Vec<float32,3> sample_color = eval_disney(base_color,
                                                      wi,
                                                      wo,
                                                      mat,
                                                      debug);

            float32 pdf = disney_pdf(wo, wi, mat, debug);

            if(pdf > 0)
            {
              sample_color = sample_color / pdf;
              data.m_throughput = data.m_throughput * sample_color;
              data.m_depth += 1;
              // advance the ray to the other side
              Vec<float32,3> hit_point = ray.m_orig + ray.m_dir * sample.m_distance;
              hit_point += eps * ray.m_dir;
              ray.m_orig = hit_point;
              ray_ptr[ii] = ray;
              data_ptr[ii] = data;
            }
            else
            {
              keep = 0;
            }


            if(debug)
            {
              std::cout<<"[shadow ray] sample "<<sample_color<<"\n";
              std::cout<<"[shadow ray] current throughput "<<data.m_throughput<<"\n";
              std::cout<<"[shadow ray] pdf "<<pdf<<"\n";
            }

          }
          else
          {
            keep = 0;
          }

          sample.m_hit_flag = keep;
          sample_ptr[ii] = sample;
        }
     }
  });

  DRAY_ERROR_CHECK();
}

void average(Array<Vec<float32,4>> &input, int32 samples)
{
  Vec<float32,4> *input_ptr = input.get_device_ptr();

  const float32 inv_samples = 1.f / float32(samples);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, input.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    Vec<float32,4> color = input_ptr[ii];
    color[0] *= inv_samples;
    color[1] *= inv_samples;
    color[2] *= inv_samples;
    color[3] = 1.f;
    input_ptr[ii] = color;

  });
  DRAY_ERROR_CHECK();
}

void process_misses(Array<Ray> &rays,
                  Array<Sample> &samples,
                  Array<RayData> &data,
                  Array<Vec<float32,4>> &colors,
                  Array<Material> &materials,
                  const float32 eps,
                  const int32 debug_ray)
{
  // add in the direct lighting for shadow rays that missed
  process_shadow_rays(rays, data, samples, colors, materials, eps, debug_ray);

  std::cout<<"Ray size "<<rays.size()<<"\n";
  std::cout<<"Samples size "<<samples.size()<<"\n";
  std::cout<<"data size "<<data.size()<<"\n";

  // remove all the misses from the q
  Array<int32> hit_flags = extract_hit_flags(samples);
  Array<int32> compact_idxs = index_flags(hit_flags);

  rays = gather(rays, compact_idxs);
  samples = gather(samples, compact_idxs);
  data = gather(data, compact_idxs);
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

void init_samples(Array<Sample> &samples)
{
  DRAY_LOG_OPEN("init_samples");

  Sample *sample_ptr = samples.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, samples.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    sample_ptr[ii].m_hit_flag = 0;
    sample_ptr[ii].m_distance = infinity32();

  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

void init_ray_data(Array<RayData> &data)
{
  DRAY_LOG_OPEN("init_ray_data");

  RayData *data_ptr = data.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, data.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    RayData data;
    constexpr Vec<float32,3> white = {{1.f,1.f,1.f}};
    data.m_throughput = white;
    data.m_flags = RayFlags::EMPTY;
    data.m_depth = 0;
    data_ptr[ii] = data;
  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

void update_samples( Array<RayHit> &hits,
                     Array<Fragment> &fragments,
                     Array<Vec<float32,4>> &colors,
                     Array<Sample> &samples,
                     const int32 mat_id)
{
  DRAY_LOG_OPEN("update_samples");

  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  const Fragment *frag_ptr = fragments.get_device_ptr_const ();

  Sample *sample_ptr = samples.get_device_ptr();
  const Vec<float32,4> *color_ptr = colors.get_device_ptr_const();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];
    Sample sample = sample_ptr[ii];
    if (hit.m_hit_idx > -1 && hit.m_dist < sample.m_distance)
    {
      sample.m_normal = frag.m_normal;
      sample.m_color  = color_ptr[ii];
      sample.m_distance = hit.m_dist;
      sample.m_hit_flag = 1;
      sample.m_mat_id = mat_id;
    }
    sample_ptr[ii] = sample;

  });
  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();
}

SphereLight test_default_light(Camera &camera, AABB<3> &bounds)
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

  //Vec<float32, 3> light_pos = pos + .1f * mag * miner_up;
  Vec<float32, 3> light_pos = pos;
  SphereLight light;
  light.m_pos = light_pos;
  light.m_radius = bounds.max_length() * 0.10;
  light.m_intensity[0] = 300.75;
  light.m_intensity[1] = 300.75;
  light.m_intensity[2] = 300.75;
  return light;
}

} // namespace detail

TestRenderer::TestRenderer()
  : m_volume(nullptr),
    m_use_lighting(true),
    m_screen_annotations(false),
    m_num_samples(10),
    m_max_depth(7),
    m_debug_ray(-1)
{
}

void TestRenderer::samples(int32 num_samples)
{
  m_num_samples = num_samples;
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
  m_sphere_lights.clear();
}

void TestRenderer::add_light(const SphereLight &light)
{
  m_sphere_lights.push_back(light);
}

void TestRenderer::add_light(const TriangleLight &light)
{
  m_tri_lights.push_back(light);
}

void TestRenderer::setup_lighting(Camera &camera)
{

  int32 max_dim = m_scene_bounds.max_dim();
  float32 radius = m_scene_bounds.m_ranges[max_dim].length() * 0.5f;
  //if(m_sphere_lights.size() + m_tri_lights.size() > 0)
  {
    m_lights.pack(m_sphere_lights, m_tri_lights, m_env_map);
  }
  //else
  //{
  //  SphereLight light = detail::test_default_light(camera, m_scene_bounds);
  //  std::vector<SphereLight> sphere_lights;
  //  sphere_lights.push_back(light);
  //  m_lights.pack(sphere_lights, m_tri_lights, m_env_map);
  //}
}

void TestRenderer::setup_materials()
{
  m_materials_array.resize(m_materials.size());
  Material *mat_ptr = m_materials_array.get_host_ptr();
  for(int32 i = 0; i < m_materials.size(); ++i)
  {
    mat_ptr[i] = m_materials[i];
  }
}

void TestRenderer::use_lighting(bool use_it)
{
  m_use_lighting = use_it;
}

void TestRenderer::add(std::shared_ptr<Traceable> traceable, Material mat)
{
  m_traceables.push_back(traceable);
  m_materials.push_back(mat);
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
      Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
      //for(int h = 0; h < rays.size(); ++h)
      //{
      //  if(rays.get_value(h).m_pixel_id == m_debug_ray)
      //  {
      //    std::cout<<"any_hit "<<hits.get_value(h).m_hit_idx
      //             <<" "<<hits.get_value(h).m_dist<<"\n";
      //  }
      //}
      detail::update_hits(hits, hit_flags);
      ray_max(rays, hits);
    }
  }

  return hit_flags;
}

Array<Sample> TestRenderer::nearest_hits(Array<Ray> &rays)
{
  const int32 size = m_traceables.size();

  Array<Sample> samples;
  samples.resize(rays.size());
  detail::init_samples(samples);

#warning "We should add a way to check the current distance so we don't do extra work or change the ray.m_far"
  for(int i = 0; i < size; ++i)
  {
    const int domains = m_traceables[i]->num_domains();
    for(int d = 0; d < domains; ++d)
    {
      m_traceables[i]->active_domain(d);
      Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
      Array<Fragment> fragments = m_traceables[i]->fragments(hits);
      //for(int h = 0; h < rays.size(); ++h)
      //{
      //  if(rays.get_value(h).m_pixel_id == m_debug_ray)
      //  {
      //    std::cout<<"nearest_hit "<<"("<<d<<") "<<hits.get_value(h).m_hit_idx
      //             <<" "<<hits.get_value(h).m_dist<<"\n";
      //  }
      //}
      int material_id = i;
      Array<Vec<float32,4>> colors;
      m_traceables[i]->colors(rays, hits, fragments, colors);
      detail::update_samples(hits, fragments, colors, samples, material_id);
    }
  }


  return samples;
}

void remove_all_but_debug(Array<Ray> &rays, int32 debug_ray)
{
  Ray ray;
  for(int h = 0; h < rays.size(); ++h)
  {
    if(rays.get_value(h).m_pixel_id == debug_ray)
    {
      ray = rays.get_value(h);
    }
  }

  rays.resize(1);
  rays.get_host_ptr()[0] = ray;
}

Framebuffer TestRenderer::render(Camera &camera)
{
  DRAY_LOG_OPEN("render");
  // init the random state. For now, just create it for each
  // pixel. In the future, we will have it live with ray specific
  // data
  m_rand_state.resize(camera.get_width() * camera.get_height());
  bool deterministic = true;
  seed_rng(m_rand_state, deterministic);

  std::vector<std::string> field_names;
  std::vector<ColorMap> color_maps;

  AABB<3> scene_bounds;
  for(int i = 0; i < m_traceables.size(); ++i)
  {
    scene_bounds.include(m_traceables[i]->collection().bounds());
  }
  m_scene_bounds = scene_bounds;


  m_scene_eps = scene_bounds.max_length() * 1e-4;


  Framebuffer framebuffer (camera.get_width(), camera.get_height());
  framebuffer.clear ();

  setup_lighting(camera);
  setup_materials();

  const int32 num_samples = m_num_samples;;
  for(int32 sample = 0; sample < num_samples; ++sample)
  {
    m_sample_count = sample;
    Array<Ray> rays;
    camera.create_rays_jitter (rays);
    //remove_all_but_debug(rays, m_debug_ray);

    Array<RayData> ray_data;
    ray_data.resize(rays.size());
    detail::init_ray_data(ray_data);

    std::cout<<"---------------  SAMPLE "<<sample<<"-------------\n";

    int iter_counter = 0;
    while(rays.size() > 0)
    {

      std::cout<<"---------------  Interation "<<iter_counter
               <<" input rays "<<rays.size()<<"-------------\n";
      Array<Sample> samples = nearest_hits(rays);

      for(int h = 0; h < rays.size(); ++h)
      {
        if(rays.get_value(h).m_pixel_id == m_debug_ray)
        {
          if(ray_data.get_value(h).m_flags == RayFlags::TRANSMITTANCE)
          {
            std::cout<<"[intersection] shadowt \n";
          }
          std::cout<<"[intersection] hit dist " <<samples.get_value(h).m_distance<<"\n";
          std::cout<<"[intersection] hit " <<samples.get_value(h).m_hit_flag<<"\n";
        }
      }

#ifdef RAY_DEBUGGING
      for(int i = 0; i < rays.size(); ++i)
      {
        if(rays.get_value(i).m_pixel_id == m_debug_ray &&
           ray_data.get_value(i).m_flags != RayFlags::TRANSMITTANCE )
        {
          RayDebug debug;
          debug.ray = rays.get_value(i);
          debug.distance = samples.get_value(i).m_distance;
          debug.hit = samples.get_value(i).m_hit_flag;
          debug.depth = ray_data.get_value(i).m_depth;
          debug.sample = m_sample_count;
          debug.shadow = 0;
          debug.red = ray_data.get_value(i).m_throughput[0];
          debug_geom[debug.ray.m_pixel_id].push_back(debug);
        }
        //if(debug.depth > 0 )
        //{
        //  std::cout<<"Debug hit "<<debug.hit<<"\n";
        //  std::cout<<"Debug distance "<<debug.distance<<"\n";
        //}
      }
#endif
      // kill rays that hit lights
      intersect_lights(rays, samples, ray_data, framebuffer);

      // reduce to only the hits
      int32 cur_size = rays.size();
      detail::process_misses(rays,
                             samples,
                             ray_data,
                             framebuffer.colors(),
                             m_materials_array,
                             m_scene_eps,
                             m_debug_ray);

      std::cout<<"[compact rays] remaining "
               <<rays.size()<<" removed "<<cur_size-rays.size()<<"\n";

      if(rays.size() == 0)
      {
        break;
      }
      std::cout<<"[compact rays]  id of one "<<rays.get_value(0).m_pixel_id<<"\n";

      // cast shadow rays
      Array<Ray> shadow_rays;
      Array<RayData> shadow_data;
      direct_lighting(rays, ray_data, samples, shadow_rays, shadow_data);

      // bounce
      bounce(rays, ray_data, samples, m_materials_array);

      // add in the shadow rays to the queue
      rays = append(rays, shadow_rays);
      ray_data = append(ray_data, shadow_data);

      // remove invalid samples and do russian roulette
      cull(ray_data, rays);

      if(m_debug_ray != -1)
      {
        for(int h = 0; h < rays.size(); ++h)
        {
          if(rays.get_value(h).m_pixel_id == m_debug_ray)
          {
            std::string type = "normal";
            if(ray_data.get_value(h).m_flags == RayFlags::TRANSMITTANCE)
            {
              type = "shadow";
            }
            std::cout<<"[throughput] "<<type<<" "<<ray_data.get_value(h).m_throughput<<"\n";
          }
        }

        std::cout<<"[current color] "<<framebuffer.colors().get_value(m_debug_ray)<<"\n";
      }
      iter_counter++;
    }
    //std::cout<<"Last ray "<<rays.get_value(0).m_pixel_id<<"\n";

  }

  detail::average(framebuffer.colors(), num_samples);


  if(m_debug_ray != -1)
    std::cout<<"[result] final color "<<framebuffer.colors().get_value(m_debug_ray)<<"\n";
  framebuffer.tone_map();
  if(m_debug_ray != -1)
    std::cout<<"[result] final color tone map"<<framebuffer.colors().get_value(m_debug_ray)<<"\n";

  // get stuff for annotations
  for(int i = 0; i < m_traceables.size(); ++i)
  {
    field_names.push_back(m_traceables[i]->field());
    color_maps.push_back(m_traceables[i]->color_map());
  }

  write_debug(framebuffer);

  if(m_screen_annotations && dray::mpi_rank() == 0)
  {
    Annotator annot;
    //framebuffer.foreground_color({{1.0f, 1.f, 1.f,1.f}});
    annot.screen_annotations(framebuffer, field_names, color_maps);
  }
  DRAY_LOG_CLOSE();

  return framebuffer;
}

void TestRenderer::bounce(Array<Ray> &rays,
                          Array<RayData> &ray_data,
                          Array<Sample> &samples,
                          Array<Material> &materials)
{
  const int32 size = rays.size();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();
  const Sample *sample_ptr = samples.get_device_ptr_const();
  Ray * ray_ptr = rays.get_device_ptr();
  RayData * data_ptr = ray_data.get_device_ptr();
  Material * mat_ptr = materials.get_device_ptr();

  const float32 eps = m_scene_eps;
  const int32 debug_ray = m_debug_ray;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    Ray ray = ray_ptr[ii];
    RayData data = data_ptr[ii];
    Sample sample = sample_ptr[ii];
    const float32 distance = sample.m_distance;
    Vec<float32,3> hit_point = ray.m_orig + ray.m_dir * distance;
    Vec<float32,3> normal = sample.m_normal;
    normal.normalize();

    if(dot(normal,-ray.m_dir) < 0)
    {
      normal = -normal;
    }

    // for now just skip the shadow rays
    if(data.m_flags == RayFlags::TRANSMITTANCE)
    {
      return;
    }

    bool debug = ray.m_pixel_id == debug_ray;
    Vec<float32,4> color = sample.m_color;
    Vec<uint32,2> rand_state = rand_ptr[ray.m_pixel_id];

    if(debug) std::cout<<"[Bounce]  in color "<<color<<"\n";


    float32 roll = randomf(rand_state);
    Material mat = mat_ptr[sample.m_mat_id];

    Vec<float32,3> wcX, wcY;
    create_basis(normal,wcX,wcY);
    Matrix<float32,3,3> to_world, to_tangent;
    to_world.set_col(0,wcX);
    to_world.set_col(1,wcY);
    to_world.set_col(2,normal);

    to_tangent = to_world.transpose();
    Vec<float32,3> wo = to_tangent * (-ray.m_dir);

    Vec<float32,3> wi;

    // TODO: stick invalid in the flags
    wi = sample_disney(wo,
                       mat,
                       data.m_flags,
                       rand_state,
                       debug);


    if(data.m_flags == RayFlags::INVALID)
    {
      color[0] = 0.f;
      color[1] = 0.f;
      color[2] = 0.f;
      color[3] = 0.f;
      wi = {{0.f,0.f,0.f}};
    }

    Vec<float32,3> sample_dir = to_world * wi;
    hit_point += eps * sample_dir;


    Vec<float32,3> base_color = {{color[0],color[1],color[2]}};

    Vec<float32,3> sample_color = eval_disney(base_color,
                                              wi,
                                              wo,
                                              mat,
                                              debug);

    data.m_pdf =  disney_pdf(wo,
                             wi,
                             mat,
                             debug);


    if(debug)
    {
      std::cout<<"[Bounce color in] "<<color<<"\n";
      std::cout<<"[Bounce sample_color] "<<sample_color<<"\n";
      std::cout<<"[Bounce pdf] "<<data.m_pdf<<"\n";
      std::cout<<"[Bounce multiplier ] "<<abs(dot(normal,sample_dir)) / data.m_pdf<<"\n";
    }



    if(data.m_pdf == 0.f)
    {
      //std::cout<<"zero pdf "<<ray.m_pixel_id<<"\n";
      // this is a bad direction
      sample_color = {{0.f, 0.f, 0.f}};
    }
    else
    {
      sample_color *= abs(dot(normal,sample_dir)) / data.m_pdf;
    }

    // attenuate sample color
    data.m_throughput = sample_color * data.m_throughput;
    // increment the depth
    data.m_depth++;

    if(debug)
    {
      std::cout<<"[Bounce color out] "<<data.m_throughput<<"\n";
      if(data.m_flags == RayFlags::INVALID) std::cout<<"[Bounce color out] invalid\n";
    }


    ray.m_dir = sample_dir;
    ray.m_orig = hit_point;
    ray.m_near = 0;
    ray.m_far = infinity<Float>();
    ray_ptr[ii] = ray;

    data_ptr[ii] = data;

    // update the random state
    rand_ptr[ray.m_pixel_id] = rand_state;
  });
}


Array<Ray>
TestRenderer::create_shadow_rays(Array<Ray> &rays,
                                 Array<RayData> &ray_data,
                                 Array<Sample> &samples,
                                 Array<Material> &materials,
                                 Array<RayData> &shadow_data)
{
  // we only want to create shadow rays for non-shadow rays
  // in the queue
  Array<int32> radiance_idxs = detail::radiance_ray_indices(ray_data);
  const int32 size = radiance_idxs.size();
  const int32 *work_idxs_ptr = radiance_idxs.get_device_ptr_const();

  Array<Ray> shadow_rays;

  shadow_rays.resize(size);
  shadow_data.resize(size);

  Sample *sample_ptr = samples.get_device_ptr();
  DeviceLightContainer d_lights(m_lights);


  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const RayData *data_ptr = ray_data.get_device_ptr_const ();
  const Material *mat_ptr = materials.get_device_ptr_const();
  RayData *shadow_data_ptr = shadow_data.get_device_ptr();

  Ray *shadow_ptr = shadow_rays.get_device_ptr();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();

  const float32 eps = m_scene_eps;
  const int32 debug_ray = m_debug_ray;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 ii = work_idxs_ptr[i];

    const Ray &ray = ray_ptr[ii];
    bool debug = ray.m_pixel_id == debug_ray;
    Sample sample = sample_ptr[ii];
    const float32 distance = sample.m_distance;
    Vec<float32,3> hit_point = ray.m_orig + ray.m_dir * distance;
    // back it away a bit
    hit_point += eps * (-ray.m_dir);

    Vec<uint32,2> rand_state = rand_ptr[ray.m_pixel_id];
    Vec<float32,3> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);
    rand[2] = randomf(rand_state);
    rand_ptr[ray.m_pixel_id] = rand_state;

    Vec<float32,3> sample_dir;
    Vec<float32,3> sample_point;
    float32 sample_distance;
    float32 light_pdf;
    Vec<float32,3> color;

    d_lights.sample(sample_dir,
                    sample_distance,
                    color,
                    light_pdf,
                    hit_point,
                    rand, debug);

    if(debug)
    {
      std::cout<<"[light sample]   dir "<<sample_dir<<"\n";
      std::cout<<"[light sample]   hit "<<hit_point<<"\n";
      std::cout<<"[light sample]   rand "<<rand<<"\n";
      std::cout<<"[light sample]   light color"<<color<<"\n";
    }

    Vec<float32,3> normal = sample.m_normal;

    if(dot(normal,-ray.m_dir) < 0)
    {
      normal = -normal;
    }

    normal.normalize();

    float32 dot_ns = dot(normal,sample_dir);
    bool valid = dot_ns > 0;

    // this used to be invalid, but now we have tranparent shadow rays
    //if(!valid)
    //{
    //  // TODO: don't cast bad ray
    //  color = {{0.f, 0.f, 0.f}};
    //}

    if(debug)
    {
      std::cout<<"[light sample]   pdf "<<light_pdf<<"\n";
      std::cout<<"[light sample]   comined "<<light_pdf<<"\n";
    }

    // do not remember why i  multiplied by alpha,m
    // but it must have somethig to do with transparency
    //float32 alpha = sample.m_color[3];

    Material mat = mat_ptr[sample_ptr[ii].m_mat_id];

    if(light_pdf > 0.f)
    {
      Vec<float32,3> base_color = {{sample.m_color[0],
                                    sample.m_color[1],
                                    sample.m_color[2]}};;
      Vec<float32,3> wcX, wcY;
      create_basis(normal,wcX,wcY);
      Matrix<float32,3,3> to_world, to_tangent;
      to_world.set_col(0,wcX);
      to_world.set_col(1,wcY);
      to_world.set_col(2,normal);
      to_tangent = to_world.transpose();

      Vec<float32,3> wo = to_tangent * (-ray.m_dir);
      Vec<float32,3> wi = to_tangent * sample_dir;

      Vec<float32,3> surface_color = eval_disney(base_color,
                                                 wi,
                                                 wo,
                                                 mat,
                                                 debug);

       float32 bsdf_pdf =  disney_pdf(wo, wi, mat, debug);

      color[0] *= surface_color[0];
      color[1] *= surface_color[1];
      color[2] *= surface_color[2];

      float32 mis_weight = power_heuristic(light_pdf, bsdf_pdf);
      color = (mis_weight * color * dot_ns) / light_pdf;
      if(debug)
      {
        std::cout<<"[light sample] mis_weight "<<mis_weight<<"\n";
        std::cout<<"[light sample] light pdf  "<<light_pdf<<"\n";
        std::cout<<"[light sample] bsdf pdf  "<<bsdf_pdf<<"\n";
        std::cout<<"[light sample] base_color "<<base_color<<"\n";
        std::cout<<"[light sample] surface_color "<<surface_color<<"\n";
        std::cout<<"[light sample] dot ns  "<<dot_ns<<"\n";
      }
    }

    color = color * data_ptr[ii].m_throughput;

    if(debug)
    {
      std::cout<<"[in thoughput "<<data_ptr[ii].m_throughput<<"\n";
      std::cout<<"[light color out "<<color<<"\n";
      std::cout<<"[light sample  dot "<<dot_ns<<"\n";
      //std::cout<<"[light sample  alpha  "<<alpha<<"\n";
    }

    Ray shadow_ray;
    shadow_ray.m_orig = hit_point;
    shadow_ray.m_dir = sample_dir;
    shadow_ray.m_near = 0.f;
    shadow_ray.m_far = sample_distance - m_scene_eps;
    shadow_ray.m_pixel_id = ray.m_pixel_id;
    shadow_ptr[i] = shadow_ray;

    if(debug)
    {
      std::cout<<"[shadow ray] dir "<<shadow_ray.m_dir<<"\n";
      std::cout<<"[shadow ray] origin "<<shadow_ray.m_orig<<"\n";
      std::cout<<"[shadow ray] far "<<shadow_ray.m_far<<"\n";
    }

    RayData data;
    data.m_throughput = color;
    data.m_depth = 0;
    data.m_flags = valid ? RayFlags::TRANSMITTANCE : RayFlags::INVALID;
    shadow_data_ptr[i] = data;
  });
#ifdef RAY_DEBUGGING
  for(int i = 0; i < shadow_rays.size(); ++i)
  {
    if(shadow_rays.get_value(i).m_pixel_id == m_debug_ray)
    {
      RayDebug debug;
      debug.ray = shadow_rays.get_value(i);
      debug.hit = 0;
      debug.shadow = 1;
      debug.sample = m_sample_count;
      debug.depth = ray_data.get_value(i).m_depth;
      debug_geom[debug.ray.m_pixel_id].push_back(debug);
    }
  }
#endif
  return shadow_rays;
}

void
TestRenderer::direct_lighting(Array<Ray> &rays,
                              Array<RayData> &ray_data,
                              Array<Sample> &samples,
                              Array<Ray> &shadow_rays,
                              Array<RayData> &shadow_data)
{
  shadow_rays = create_shadow_rays(rays,
                                   ray_data,
                                   samples,
                                   m_materials_array,
                                   shadow_data);
}

void TestRenderer::intersect_lights(Array<Ray> &rays,
                                    Array<Sample> &samples,
                                    Array<RayData> &data,
                                    Framebuffer &fb)
{
  Sample *sample_ptr = samples.get_device_ptr();
  const RayData *data_ptr = data.get_device_ptr_const();
  const Ray *ray_ptr = rays.get_device_ptr_const ();

  DeviceLightContainer d_lights(m_lights);

  Vec<float32,4> *color_ptr = fb.colors().get_device_ptr();

  const int32 debug_ray = m_debug_ray;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray ray = ray_ptr[ii];
    bool debug = ray.m_pixel_id == debug_ray;
    Sample sample = sample_ptr[ii];
    RayData data = data_ptr[ii];
    float32 light_pdf = 0;
    Vec<float32,3> light_radiance = {{0.f,0.f,0.f}};;

    if(data.m_flags == RayFlags::TRANSMITTANCE)
    {
      // this needs to change
      return;
    }

    float32 nearest_dist = sample.m_distance;
    int32 hit = sample.m_hit_flag;
    if(hit != 1)
    {
      nearest_dist = infinity32();
    }

    light_radiance = d_lights.intersect(ray, nearest_dist, light_pdf);

    if(!is_black(light_radiance))
    {
      // Kill this ray
      sample.m_hit_flag = 0;

      if(data.m_depth > 0 && (data.m_flags & RayFlags::DIFFUSE))
      {
        // this was a diffuse bounce, so mix the light sample
        // with the diffuse pdf
        // the pdf which generated the ray direction goes first.
        //
        // In practice, if you generated samples from a whole bunch of different
        // strategies, the power heuristic would handle an arbitrary number of
        // parameters, e.g. if you took samples with three strategies,
        // then the weight for the first strategy would be
        // (f * f) / (f * f + g * g + h * h),
        // the weight for the second (g * g) / (f * f + g * g + h * h), etc.
        // In this case, we have sampled the diffuse brdf, but we also sampled
        // the direct lighting. Thus we need to weight the diffuse pdf to account
        // for the direct hit to the light and the other sample
        light_radiance = power_heuristic(data.m_pdf, light_pdf) * light_radiance;
        if(debug)
        {
          std::cout<<"[intersect lights] diffuse light hit \n";
          std::cout<<"[intersect lights]         light pdf "<<light_pdf<<"\n";
          std::cout<<"[intersect lights]         data pdf  "<<data.m_pdf<<"\n";
          float32 temp = power_heuristic(data.m_pdf, light_pdf);
          std::cout<<"[intersect lights]         hueristic  "<<temp<<"\n";
        }
      }

      if(debug)
      {
        std::cout<<"[intersect lights] light dist "<<nearest_dist<<"\n";
        Vec<float32,3> contrib;
        contrib[0] = light_radiance[0] * data.m_throughput[0];
        contrib[1] = light_radiance[1] * data.m_throughput[1];
        contrib[2] = light_radiance[2] * data.m_throughput[2];
        std::cout<<"[intersect lights] contribution "<<contrib<<"\n";
      }



      color_ptr[ray.m_pixel_id][0] += light_radiance[0] * data.m_throughput[0];
      color_ptr[ray.m_pixel_id][1] += light_radiance[1] * data.m_throughput[1];
      color_ptr[ray.m_pixel_id][2] += light_radiance[2] * data.m_throughput[2];
      color_ptr[ray.m_pixel_id][3] = 1.f;
    }
    sample_ptr[ii] = sample;

  });
  DRAY_ERROR_CHECK();
}

void TestRenderer::cull(Array<RayData> &data,
                        Array<Ray> &rays)
{
  Array<int32> keep_flags;
  keep_flags.resize(rays.size());
  int32 *keep_ptr = keep_flags.get_device_ptr();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();

  RayData *data_ptr = data.get_device_ptr();
  const Ray *ray_ptr = rays.get_device_ptr_const();
  const int32 max_depth = m_max_depth;
  const int32 debug_ray = m_debug_ray;


  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    int32 keep = 1;
    RayData data = data_ptr[ii];
    Vec<float32,3> att = data.m_throughput;
    float32 max_att = max(att[0], max(att[1], att[2]));
    int32 pixel_id = ray_ptr[ii].m_pixel_id;
    bool debug = pixel_id == debug_ray;

    if(data.m_flags == RayFlags::INVALID || data.m_depth > max_depth)
    {
      // cull invalid samples
      keep = 0;
    }

    // TODO: is this how we should treat transmittance (shadow) rays
    // russian roulette
    if(data.m_depth > 3 && data.m_flags != RayFlags::TRANSMITTANCE)
    {
      Vec<uint32,2> rand_state = rand_ptr[pixel_id];
      float32 roll = randomf(rand_state);
      rand_ptr[pixel_id] = rand_state;

      // I think theshold should be calculated
      // by the max potential contribution, that is
      // max_val * max_light_power
      float32 threshold = 0.1;
      if(max_att < threshold)
      {
        float32 q = max(0.05f, 1.f - max_att);
        if(roll < q)
        {
          // kill
          keep = 0;
        }

        att *= 1.f/(1. - q);

        if(debug && keep == 1)
        {
          std::cout<<"[cull] attenuation correction "<<(1.f/(1. - q))
                   <<" roll "<<roll<<" q "<<q<<" max att "<<max_att<<"\n";
        }
        else if(debug)
        {
          std::cout<<"[cull] q "<<q<<"\n";
          std::cout<<"[cull] roll "<<q<<"\n";
          std::cout<<"[cull] throughput "<<data.m_throughput<<"\n";
        }
      }
    }

    if(debug)
    {
      std::cout<<"[cull] keep "<<keep<<"\n";
    }


    data.m_throughput = att;
    data_ptr[ii] = data;
    keep_ptr[ii] = keep;
  });

  DRAY_ERROR_CHECK();

  int32 before_size = rays.size();

  Array<int32> compact_idxs = index_flags(keep_flags);

  std::cout<<"## rays "<<rays.size()<<"\n";
  std::cout<<"## data "<<data.size()<<"\n";

  rays = gather(rays, compact_idxs);
  data = gather(data, compact_idxs);

  std::cout<<"[cull] removed "<<before_size - rays.size()<<"\n";
}

void TestRenderer::write_debug(Framebuffer &fb)
{
  if(m_debug_ray == -1)
  {
    return;
  }

  conduit::Node fb_dataset;
  conduit::Node &buff = fb_dataset.append();
  fb.to_node(buff);
  conduit::relay::io_blueprint::save(buff, "pixel_debugging.blueprint_root");

#ifdef RAY_DEBUGGING
  conduit::Node domain;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<int32> conn;
  std::vector<float> pixel_ids;
  std::vector<float> depths;
  std::vector<float> samples;
  std::vector<float> hits;
  std::vector<float> shadow;
  std::vector<float> red;
  std::vector<float> final_red;

  float default_dist = m_scene_bounds.max_length() / 4.f;

  std::cout<<"Debug pixels "<<debug_geom.size()<<"\n";

  std::cout<<"Debug color "<<fb.colors().get_value(m_debug_ray)<<"\n";
  int conn_count = 0;
  for(auto &ray : debug_geom)
  {
    std::vector<RayDebug> &debug = ray.second;
    int32 pixel_id = ray.first;

    for(int i = 0; i < debug.size(); ++i)
    {
      Ray ray = debug[i].ray;
      //if(fb.colors().get_value(ray.m_pixel_id)[0] == 0.f)
      //{
      //  std::cout<<ray.m_pixel_id<<" ";
      //}
      if(i == 0 && debug[i].hit != 1)
      {
        // get rid of all rays that initially miss
        break;;
      }
      // only write out the debug ray;
      if(ray.m_pixel_id != m_debug_ray)
      {
        break;
      }
      conn.push_back(conn_count);
      conn_count++;
      conn.push_back(conn_count);
      conn_count++;

      pixel_ids.push_back(pixel_id);
      final_red.push_back(fb.colors().get_value(pixel_id)[0]);
      depths.push_back(debug[i].depth);
      samples.push_back(debug[i].sample);
      x.push_back(ray.m_orig[0]);
      y.push_back(ray.m_orig[1]);
      z.push_back(ray.m_orig[2]);
      shadow.push_back(debug[i].shadow);
      red.push_back(debug[i].red);
      Vec<float32,3> end;
      if(debug[i].hit && debug[i].distance > m_scene_eps)
      {
        end = ray.m_orig + ray.m_dir * debug[i].distance;
        hits.push_back(1.0f);
      }
      else if(debug[i].shadow == 1)
      {
        end = ray.m_orig + ray.m_dir * ray.m_far;
        hits.push_back(debug[i].hit);
      }
      else
      {
        end = ray.m_orig + ray.m_dir * default_dist;
        hits.push_back(0.0f);
      }
      if(end[0] == infinity32() ||
         end[1] == infinity32() ||
         end[2] == infinity32() ||
         end[0] == -infinity32() ||
         end[1] == -infinity32() ||
         end[2] == -infinity32() )
      {
        end = ray.m_orig + ray.m_dir * default_dist;
      }
      //std::cout<<end<<"\n";
      x.push_back(end[0]);
      y.push_back(end[1]);
      z.push_back(end[2]);
    }

  }

  domain["coordsets/coords/type"] = "explicit";
  domain["coordsets/coords/values/x"].set(x);
  domain["coordsets/coords/values/y"].set(y);
  domain["coordsets/coords/values/z"].set(z);
  domain["topologies/mesh/type"] = "unstructured";
  domain["topologies/mesh/coordset"] = "coords";
  domain["topologies/mesh/elements/shape"] = "line";
  domain["topologies/mesh/elements/connectivity"].set(conn);

  domain["fields/pixel/association"] = "element";
  domain["fields/pixel/topology"] = "mesh";
  domain["fields/pixel/values"].set(pixel_ids);

  domain["fields/hits/association"] = "element";
  domain["fields/hits/topology"] = "mesh";
  domain["fields/hits/values"].set(hits);

  domain["fields/depth/association"] = "element";
  domain["fields/depth/topology"] = "mesh";
  domain["fields/depth/values"].set(depths);

  domain["fields/shadow/association"] = "element";
  domain["fields/shadow/topology"] = "mesh";
  domain["fields/shadow/values"].set(shadow);

  domain["fields/red/association"] = "element";
  domain["fields/red/topology"] = "mesh";
  domain["fields/red/values"].set(red);

  domain["fields/final_red/association"] = "element";
  domain["fields/final_red/topology"] = "mesh";
  domain["fields/final_red/values"].set(final_red);

  domain["fields/sample/association"] = "element";
  domain["fields/sample/topology"] = "mesh";
  domain["fields/sample/values"].set(samples);

  conduit::Node dataset;
  dataset.append() = domain;
  conduit::Node info;
  if(!conduit::blueprint::mesh::verify(dataset,info))
  {
    info.print();
  }
  conduit::relay::io_blueprint::save(domain, "ray_debugging.blueprint_root");
#endif
}

void TestRenderer::load_env_map(const std::string filename)
{
  m_env_map.load(filename);
}

} // namespace dray

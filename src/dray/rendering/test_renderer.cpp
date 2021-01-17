// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/test_renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/rendering/annotator.hpp>
#include <dray/rendering/low_order_intersectors.hpp>
#include <dray/rendering/sampling.hpp>
#include <dray/rendering/disney_sampling.hpp>
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
int debug_ray = 0;
int zero_count = 0;
int total_count = 0;

namespace dray
{

static float32 ray_eps = 1e-5;

namespace detail
{


// the pdf which generated the ray direction goes first.
DRAY_EXEC
float32 power_heuristic(float32 a, float32 b)
{
  float t = a * a;
  return t / (b * b + t);
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

void add(Array<Ray> &rays,
         Array<Vec<float32,3>> &input,
         Array<Vec<float32,4>> &output,
         Array<RayData> &data)
{
  Vec<float32,4> *output_ptr = output.get_device_ptr();
  const Vec<float32,3> *input_ptr = input.get_device_ptr_const();
  const RayData *data_ptr = data.get_device_ptr_const();
  const Ray *ray_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, input.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const int32 idx = ray_ptr[ii].m_pixel_id;
    Vec<float32,4> color = output_ptr[idx];
    // FINDME TODO
    Vec<float32,3> in = input_ptr[ii];
    if(idx == debug_ray) std::cout<<"[add] "
                                  <<in<<" current thoughput"<<data_ptr[ii].m_throughput<<"\n";

    color[0] += in[0];
    color[1] += in[1];
    color[2] += in[2];
    color[3] = 1.f;
    output_ptr[idx] = color;
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
    // TODO: transparency
    //color[3] = 1;
    //color[0] = clamp(color[0],0.f,1.f);
    //color[1] = clamp(color[1],0.f,1.f);
    //color[2] = clamp(color[2],0.f,1.f);
    input_ptr[ii] = color;

  });
  DRAY_ERROR_CHECK();
}

void compact_hits(Array<Ray> &rays, Array<Sample> &samples, Array<RayData> &data)
{
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
    data.m_is_specular = false;
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
    // could check the current distance
    // but we adjust the ray max do this shouldn't need to
    if (hit.m_hit_idx > -1)
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
    m_num_samples(10)
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

  if(m_sphere_lights.size() + m_tri_lights.size() > 0)
  {
    m_lights.pack(m_sphere_lights, m_tri_lights);
  }
  else
  {
    SphereLight light = detail::test_default_light(camera, m_scene_bounds);
    std::vector<SphereLight> sphere_lights;
    sphere_lights.push_back(light);
    m_lights.pack(sphere_lights, m_tri_lights);
  }
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
      // TODO: actually make and any hit method
      Array<RayHit> hits = m_traceables[i]->nearest_hit(rays);
      //for(int h = 0; h < rays.size(); ++h)
      //{
      //  if(rays.get_value(h).m_pixel_id == debug_ray)
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
      //  if(rays.get_value(h).m_pixel_id == debug_ray)
      //  {
      //    std::cout<<"nearest_hit "<<"("<<d<<") "<<hits.get_value(h).m_hit_idx
      //             <<" "<<hits.get_value(h).m_dist<<"\n";
      //  }
      //}
      int material_id = i;
      Array<Vec<float32,4>> colors;
      m_traceables[i]->colors(rays, hits, fragments, colors);
      detail::update_samples(hits, fragments, colors, samples, material_id);
      ray_max(rays, hits);
    }
  }

  for(int h = 0; h < rays.size(); ++h)
  {
    if(rays.get_value(h).m_pixel_id == debug_ray)
    {
      std::cout<<"[intersection] hit dist " <<samples.get_value(h).m_distance<<"\n";
      std::cout<<"[intersection] hit " <<samples.get_value(h).m_hit_flag<<"\n";
    }
  }

  return samples;
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


  ray_eps = scene_bounds.max_length() * 1e-4;

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

    int32 max_depth = 8;
    m_depth = 0;

    Array<RayData> ray_data;
    ray_data.resize(rays.size());
    detail::init_ray_data(ray_data);

    for(int32 depth = 0; depth < max_depth; ++depth)
    {
      m_depth = depth;

      std::cout<<"--------------- Depth "<<depth
               <<" input rays "<<rays.size()<<"-------------\n";
      Array<Sample> samples = nearest_hits(rays);
#ifdef RAY_DEBUGGING
      for(int i = 0; i < rays.size(); ++i)
      {
        RayDebug debug;
        debug.ray = rays.get_value(i);
        debug.distance = samples.get_value(i).m_distance;
        debug.hit = samples.get_value(i).m_hit_flag;
        debug.depth = m_depth;
        debug.sample = m_sample_count;
        debug.shadow = 0;
        debug.red = ray_data.get_value(i).m_throughput[0];
        debug_geom[debug.ray.m_pixel_id].push_back(debug);
        //if(debug.depth > 0 )
        //{
        //  std::cout<<"Debug hit "<<debug.hit<<"\n";
        //  std::cout<<"Debug distance "<<debug.distance<<"\n";
        //}
      }
#endif
      // kill rays that hit lights and don't add the colors,
      // since that would be double sampling the lights
      intersect_lights(rays, samples, ray_data, framebuffer, depth);

      // reduce to only the hits
      int32 cur_size = rays.size();
      detail::compact_hits(rays, samples, ray_data);
      std::cout<<"[compact rays] remaining "
               <<rays.size()<<" removed "<<cur_size-rays.size()<<"\n";

      if(rays.size() == 0)
      {
        break;
      }
      std::cout<<"[compact rays]  id of one "<<rays.get_value(0).m_pixel_id<<"\n";

      // cast shadow rays
      Array<Vec<float32,3>> light_colors = direct_lighting(rays, samples);
      // add in the attenuation
      detail::multiply(light_colors, ray_data);
      // add the light contribution
      detail::add(rays, light_colors, framebuffer.colors(), ray_data);

      // bounce
      bounce(rays, ray_data, samples, m_materials_array);

      // attenuate the light
      detail::multiply(ray_data, samples);

      if(depth >= 3)
      {
        russian_roulette(ray_data, samples, rays);
        if(rays.size() == 0)
        {
          break;
        }
      }

      for(int h = 0; h < rays.size(); ++h)
      {
        if(rays.get_value(h).m_pixel_id == debug_ray)
        {
          std::cout<<"[throughput] "<<ray_data.get_value(h).m_throughput<<"\n";
        }
      }

      std::cout<<"[current color] "<<framebuffer.colors().get_value(debug_ray)<<"\n";
    }
    //std::cout<<"Last ray "<<rays.get_value(0).m_pixel_id<<"\n";

  }

  std::cout<<"Zero pdf percent "<<float32(zero_count) / float32(total_count)<<"\n";

  detail::average(framebuffer.colors(), num_samples);


  std::cout<<"[result] final color "<<framebuffer.colors().get_value(debug_ray)<<"\n";
  framebuffer.tone_map();
  std::cout<<"[result] final color tone map"<<framebuffer.colors().get_value(debug_ray)<<"\n";

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
  Sample *sample_ptr = samples.get_device_ptr();
  Ray * ray_ptr = rays.get_device_ptr();
  RayData * data_ptr = ray_data.get_device_ptr();
  Material * mat_ptr = materials.get_device_ptr();

  const float32 eps = ray_eps;

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

    wi = sample_disney(wo,
                       mat,
                       data.m_is_specular,
                       rand_state,
                       debug);

    Vec<float32,3> sample_dir = to_world * wi;
    hit_point += eps * sample_dir;

    Vec<float32,3> base_color = {{color[0],color[1],color[1]}};

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

    total_count++;
    if(data.m_pdf == 0.f)
    {
      //std::cout<<"zero pdf "<<ray.m_pixel_id<<"\n";
      zero_count++;
      // this is a bad direction
      sample_color = {{0.f, 0.f, 0.f}};
    }
    else
    {
      sample_color *= abs(dot(normal,sample_dir)) / data.m_pdf;
    }

    color[0] = sample_color[0];
    color[1] = sample_color[1];
    color[2] = sample_color[2];

    if(debug)
    {
      std::cout<<"[Bounce color out] "<<color<<"\n";
    }

    ray.m_dir = sample_dir;
    ray.m_orig = hit_point;
    ray.m_near = 0;
    ray.m_far = infinity<Float>();
    ray_ptr[ii] = ray;
    data_ptr[ii] = data;

    // update the random state
    rand_ptr[ray.m_pixel_id] = rand_state;
    sample.m_color = color;
    sample_ptr[ii] = sample;
  });
}


Array<Ray>
TestRenderer::create_shadow_rays(Array<Ray> &rays,
                                 Array<Sample> &samples,
                                 Array<Vec<float32,3>> &light_colors,
                                 Array<Material> &materials)
{
  Array<Ray> shadow_rays;
  shadow_rays.resize(rays.size());
  light_colors.resize(rays.size());

  Sample *sample_ptr = samples.get_device_ptr();
  DeviceLightContainer d_lights(m_lights);

  // create a uniform sampleing of lights
  // TODO: weight by power


  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const Material *mat_ptr = materials.get_device_ptr_const();

  Vec<float32,3> *color_ptr = light_colors.get_device_ptr();
  Ray *shadow_ptr = shadow_rays.get_device_ptr();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();

  const float32 eps = ray_eps;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray &ray = ray_ptr[ii];
    bool debug = ray.m_pixel_id == debug_ray;
    Sample sample = sample_ptr[ii];
    const float32 distance = sample.m_distance;
    Vec<float32,3> hit_point = ray.m_orig + ray.m_dir * distance;
    // back it away a bit
    hit_point += eps * (-ray.m_dir);

    Vec<uint32,2> rand_state = rand_ptr[ray.m_pixel_id];
    Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);

    float32 sample_pdf;
    int32 light_idx = d_lights.m_distribution.discrete_sample(randomf(rand_state),sample_pdf, debug);
    // update the random state
    rand_ptr[ray.m_pixel_id] = rand_state;

    Vec<float32,3> sample_dir;
    Vec<float32,3> sample_point;
    float32 light_pdf;
    Vec<float32,3> light_normal;
    Vec<float32,3> color;

    if(d_lights.m_types[light_idx] == LightType::sphere)
    {
      const SphereLight light = d_lights.sphere_light(light_idx);
      sample_point = light.sample(hit_point, rand, light_pdf, debug);
      sample_dir = sample_point - hit_point;
      light_normal = sample_point - light.m_pos;
      color = light.m_intensity;
      // this point was chosen with respect to the solid angle,
      // so we know its facing the right way
    }
    else
    {
      // triangle
      const TriangleLight light = d_lights.triangle_light(light_idx);
      sample_point = light.sample(hit_point, rand, light_pdf, debug);
      sample_dir = sample_point - hit_point;
      color = light.m_intensity;

      light_normal = cross(light.m_v1 - light.m_v0,
                           light.m_v2 - light.m_v0);
      if(dot(light_normal,sample_dir) > 0)
      {
        light_normal = -light_normal;
      }
    }

    float32 sample_distance = sample_dir.magnitude();

    sample_dir.normalize();
    light_normal.normalize();
    float32 cos_light = dot(light_normal,-sample_dir);

    if(debug)
    {
      std::cout<<"[light sample]   dir "<<sample_dir<<"\n";
      std::cout<<"[light sample]   hit "<<hit_point<<"\n";
      std::cout<<"[light sample]   distance "<<sample_distance<<"\n";
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

    if(!valid)
    {
      // TODO: don't cast bad ray
      color = {{0.f, 0.f, 0.f}};
    }

    // the pdf output by the light is something like 1/area
    const float32 solid_angle_pdf = (sample_distance*sample_distance) / cos_light;
    light_pdf *= solid_angle_pdf * sample_pdf;
    if(debug)
    {
      std::cout<<"[light sample]   pdf "<<light_pdf<<"\n";
      std::cout<<"[light sample]   sample_pdf "<<sample_pdf<<"\n";
      std::cout<<"[light sample]   solid_angle_pdf "<<solid_angle_pdf<<"\n";
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
                                                 mat);
//      Vec<float32,3> surface_color = eval_color(normal,
//                                                sample_dir,
//                                                -ray.m_dir,
//                                                base_color,
//                                                mat.m_roughness,
//                                                mat.m_diff_ratio, debug);

       float32 bsdf_pdf =  disney_pdf(wo, wi, mat, debug);

      //float32 bsdf_pdf = eval_pdf(sample_dir,
      //                            -ray.m_dir,
      //                            normal,
      //                            mat.m_roughness,
      //                            mat.m_diff_ratio);

      color[0] *= surface_color[0];
      color[1] *= surface_color[1];
      color[2] *= surface_color[2];

      float32 mis_weight = detail::power_heuristic(light_pdf, bsdf_pdf);
      color = (mis_weight * color * dot_ns)/ light_pdf;
      if(debug)
      {
        std::cout<<"[light sample] mis_weight "<<mis_weight<<"\n";
        std::cout<<"[light sample] light pdf  "<<light_pdf<<"\n";
        std::cout<<"[light sample] bsdf pdf  "<<bsdf_pdf<<"\n";
        std::cout<<"[light sample] base_color "<<base_color<<"\n";
        std::cout<<"[light sample] surface_color "<<surface_color<<"\n";
      }
    }

    if(debug)
    {
      std::cout<<"[light color out "<<color<<"\n";
      std::cout<<"[light sample  dot "<<dot_ns<<"\n";
      //std::cout<<"[light sample  alpha  "<<alpha<<"\n";
    }

    color_ptr[ii] = color;

    Ray shadow_ray;
    shadow_ray.m_orig = hit_point;
    shadow_ray.m_dir = sample_dir;
    shadow_ray.m_near = 0.f;
    shadow_ray.m_far = sample_distance;
    shadow_ray.m_pixel_id = ray.m_pixel_id;
    shadow_ptr[ii] = shadow_ray;
  });
#ifdef RAY_DEBUGGING
  for(int i = 0; i < shadow_rays.size(); ++i)
  {
    RayDebug debug;
    debug.ray = shadow_rays.get_value(i);;
    debug.hit = 0;
    debug.shadow = 1;
    debug.sample = m_sample_count;
    debug.depth = m_depth;
    debug_geom[debug.ray.m_pixel_id].push_back(debug);
  }
#endif
  return shadow_rays;
}



void
TestRenderer::shade_lights(Array<int32> &hit_flags,
                           Array<Vec<float32,3>> &light_colors,
                           Array<Vec<float32,3>> &colors)
{
  const int32 *hit_flag_ptr = hit_flags.get_device_ptr_const();

  const Vec<float32,3> *light_color_ptr = light_colors.get_device_ptr_const();
  Vec<float32,3> *color_ptr = colors.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hit_flags.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    Vec<float32,3> in_color = light_color_ptr[ii];

    if(hit_flag_ptr[ii] != 0)
    {
      in_color *= 0.f;
    }
    color_ptr[ii] += in_color;

  });
}

Array<Vec<float32,3>>
TestRenderer::direct_lighting(Array<Ray> &rays,
                              Array<Sample> &samples)
{
  Array<Vec<float32,3>> contributions;
  contributions.resize(rays.size());
  Vec<float32,3> black = {{0.f, 0.f, 0.f}};
  array_memset_vec (contributions, black);

  Array<Vec<float32,3>> light_colors;

  Array<Ray> shadow_rays = create_shadow_rays(rays,
                                                samples,
                                                light_colors,
                                                m_materials_array);

  Array<int32> hit_flags = any_hit(shadow_rays);

  shade_lights(hit_flags,
               light_colors,
               contributions);

  return contributions;

}

void TestRenderer::intersect_lights(Array<Ray> &rays,
                                    Array<Sample> &samples,
                                    Array<RayData> &data,
                                    Framebuffer &fb,
                                    int32 depth)
{
  Sample *sample_ptr = samples.get_device_ptr();
  const RayData *data_ptr = data.get_device_ptr_const();
  const Ray *ray_ptr = rays.get_device_ptr_const ();

  DeviceLightContainer d_lights(m_lights);
  const int32 num_lights = m_lights.m_num_lights;

  Vec<float32,4> *color_ptr = fb.colors().get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray ray = ray_ptr[ii];
    int32 light_id = -1;
    bool debug = ray.m_pixel_id == debug_ray;
    Sample sample = sample_ptr[ii];
    RayData data = data_ptr[ii];
    float32 light_pdf = 0;

    float32 nearest_dist = sample.m_distance;
    int32 hit = sample.m_hit_flag;
    if(hit != 1)
    {
      nearest_dist = infinity32();
    }

    for(int32 i = 0; i < num_lights; ++i)
    {
      float32 dist;
      float32 temp_pdf;
      if(d_lights.m_types[i] == LightType::sphere)
      {
        const SphereLight light = d_lights.sphere_light(i);
        dist = light.intersect(ray.m_orig, ray.m_dir, temp_pdf);
      }
      else
      {
        // triangle
        const TriangleLight light = d_lights.triangle_light(i);
        dist = light.intersect(ray.m_orig, ray.m_dir, temp_pdf);
      }

      if(dist < nearest_dist && dist < ray.m_far && dist >= ray.m_near)
      {
        light_id = i;
        nearest_dist = dist;
        light_pdf = temp_pdf;
      }
    }

    if(light_id != -1)
    {
      // Kill this ray
      sample.m_hit_flag = 0;
      Vec<float32,3> radiance = d_lights.intensity(light_id);

      if(depth > 0 && !data.m_is_specular)
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
        radiance = detail::power_heuristic(data.m_pdf, light_pdf) * radiance;
        if(debug)
        {
          std::cout<<"[intersect lights] diffuse light hit \n";
          std::cout<<"[intersect lights]         light pdf "<<light_pdf<<"\n";
          std::cout<<"[intersect lights]         data pdf  "<<data.m_pdf<<"\n";
          float32 temp = detail::power_heuristic(data.m_pdf, light_pdf);
          std::cout<<"[intersect lights]         hueristic  "<<temp<<"\n";
        }
      }

      if(debug)
      {
        std::cout<<"[intersect lights] hit light "<<light_id<<"\n";
        std::cout<<"[intersect lights] light dist "<<nearest_dist<<"\n";
        Vec<float32,3> contrib;
        contrib[0] = radiance[0] * data.m_throughput[0];
        contrib[1] = radiance[1] * data.m_throughput[1];
        contrib[2] = radiance[2] * data.m_throughput[2];
        std::cout<<"[intersect lights] contribution "<<contrib<<"\n";
      }

      color_ptr[ray.m_pixel_id][0] += radiance[0] * data.m_throughput[0];
      color_ptr[ray.m_pixel_id][1] += radiance[1] * data.m_throughput[1];
      color_ptr[ray.m_pixel_id][2] += radiance[2] * data.m_throughput[2];
      color_ptr[ray.m_pixel_id][3] = 1.f;
    }
    sample_ptr[ii] = sample;

  });
  DRAY_ERROR_CHECK();
}

void TestRenderer::russian_roulette(Array<RayData> &data,
                                    Array<Sample> &samples,
                                    Array<Ray> &rays)
{
  Array<int32> keep_flags;
  keep_flags.resize(rays.size());
  int32 *keep_ptr = keep_flags.get_device_ptr();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();

  RayData *data_ptr = data.get_device_ptr();
  const Ray *ray_ptr = rays.get_device_ptr_const();


  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    int32 keep = 1;
    RayData data = data_ptr[ii];
    Vec<float32,3> att = data.m_throughput;
    float32 max_att = max(att[0], max(att[1], att[2]));
    int32 pixel_id = ray_ptr[ii].m_pixel_id;
    bool debug = pixel_id == debug_ray;
    Vec<uint32,2> rand_state = rand_ptr[pixel_id];

    float32 roll = randomf(rand_state);
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
    }

    if(debug)
    {
      std::cout<<"[cull] keep "<<keep<<"\n";
    }
    data.m_throughput = att;
    data_ptr[ii] = data;
    keep_ptr[ii] = keep;
    rand_ptr[pixel_id] = rand_state;
  });

  DRAY_ERROR_CHECK();

  int32 before_size = rays.size();

  Array<int32> compact_idxs = index_flags(keep_flags);

  rays = gather(rays, compact_idxs);
  samples = gather(samples, compact_idxs);
  data = gather(data, compact_idxs);

  std::cout<<"[cull] removed "<<before_size - rays.size()<<"\n";
}

void TestRenderer::write_debug(Framebuffer &fb)
{

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

  std::cout<<"Debug color "<<fb.colors().get_value(debug_ray)<<"\n";
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
      if(ray.m_pixel_id != debug_ray)
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
      if(debug[i].hit && debug[i].distance > ray_eps)
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

} // namespace dray

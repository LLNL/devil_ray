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

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <memory>
#include <vector>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

#define RAY_DEBUGGING
int debug_ray = 83060;
namespace dray
{

static float32 ray_eps = 1e-5;

namespace detail
{
DRAY_EXEC
Vec3f reflect(const Vec3f &i, const Vec3f &n)
{
  return i - 2.f * dot(i, n) * n;
}


DRAY_EXEC
float32 fresnel_reflectance(float ior_outside,
                            float ior_inside,
                            Vec3f normal,
                            Vec3f incident,
                            float specular_chance,
                            float f90 = 1.f)
{
  // Schlick aproximation
  float r0 = (ior_outside-ior_inside) / (ior_outside+ior_inside);
  r0 *= r0;
  float cosX = -dot(normal, incident);
  if (ior_outside > ior_inside)
  {
      float n = ior_inside/ior_outside;
      float sinT2 = n*n*(1.0-cosX*cosX);
      // Total internal reflection
      if (sinT2 > 1.0)
          return f90;
      cosX = sqrt(1.0-sinT2);
  }
  float x = 1.0 - cosX;
  float ret = r0+(1.0-r0)*x*x*x*x*x;

  // adjust reflect multiplier for object reflectivity
  return lerp(specular_chance, f90, ret);
}

DRAY_EXEC
float32 intersect_sphere(const Vec<float32,3> &center,
                         const float32 &radius,
                         const Vec<float32,3> &origin,
                         const Vec<float32,3> &dir)
{
  float32 dist = infinity32();

  Vec<float32, 3> l = center - origin;

  float32 dot1 = dot(l, dir);
  if (dot1 >= 0)
  {
    float32 d = dot(l, l) - dot1 * dot1;
    float32 r2 = radius * radius;
    if (d <= r2)
    {
      float32 tch = sqrt(r2 - d);
      dist = dot1 - tch;
    }
  }
  return dist;
}


DRAY_EXEC
void create_basis(const Vec<Float, 3> &normal,
                   Vec<Float, 3> &xAxis,
                   Vec<Float, 3> &yAxis)
{
  // generate orthoganal basis about normal (i.e. basis for tangent space).
  // kz will be the axis idx (0,1,2) most aligned with normal.
  // TODO MAI [2018-05-30] I propose we instead choose the axis LEAST aligned with normal;
  // this amounts to flipping all the > to instead be <.
  int32 kz = 0;
  if (fabs (normal[0]) > fabs (normal[1]))
  {
    if (fabs (normal[0]) > fabs (normal[2]))
      kz = 0;
    else
      kz = 2;
  }
  else
  {
    if (fabs (normal[1]) > fabs (normal[2]))
      kz = 1;
    else
      kz = 2;
  }
  // nonNormal will be the axis vector most aligned with normal. (future: least aligned?)
  Vec<Float, 3> notNormal;
  notNormal[0] = 0.f;
  notNormal[1] = 0.f;
  notNormal[2] = 0.f;
  notNormal[(kz + 1) % 3] = 1.f; //[M.A.I. 5/31]

  xAxis = cross (normal, notNormal);
  xAxis.normalize ();
  yAxis = cross (normal, xAxis);
  yAxis.normalize ();
}

DRAY_EXEC
Vec<Float, 3>
cosine_weighted_hemisphere ( const Vec<float32,3> &normal,
                             const Vec<float32,2> &xy,
                             float32 &test_out)
{
  const float32 phi = 2.f * pi() * xy[0];
  const float32 cosTheta = sqrt(xy[1]), sinTheta = sqrt(1.0f - xy[1]);
  Vec<float32, 3> direction;
  direction[0] = cos(phi);
  direction[1] = sin(phi) * sinTheta;
  direction[2] = cosTheta;

  test_out = cosTheta;
  //const float32 r = sqrt (xy[0]);
  //const float32 theta = 2 * pi () * xy[1];

  //Vec<float32, 3> direction;
  //direction[0] = r * cos (theta);
  //direction[1] = r * sin (theta);
  //direction[2] = sqrt (max (0.0f, 1.f - xy[0]));

  // transform the direction into the normals orientation
  Vec<Float, 3> tangent_x, tangent_y;
  create_basis(normal, tangent_x, tangent_y);

  Vec<Float, 3> sample_dir = tangent_x * direction[0] +
                             tangent_y * direction[1] +
                             normal * direction[2];

  return sample_dir;
}

DRAY_EXEC
Vec3f phong_brdf(const Vec3f &dir,
                 const Vec3f &normal,
                 const float32 &shine,
                 const Vec<float32,2> &u,
                 float32 &pdf,
                 bool debug = false)
{

  float32 alpha = acos(pow(u[0], 1.f/(1.f+shine)));
  float32 phi = 2.f * pi() * u[1];
  Vec3f local_dir;
  local_dir[0] = sin(alpha) * cos(phi);
  local_dir[1] = sin(alpha) * sin(phi);
  local_dir[2] = cos(alpha);


  Vec3f reflect_dir = reflect(dir, normal);

  if(debug)
  {
    std::cout<<"Rand "<<u<<"\n";
    std::cout<<"alpha "<<alpha<<" phi "<<phi<<"\n";
    std::cout<<"local_dir "<<local_dir<<"\n";
    std::cout<<"reflect "<<reflect_dir<<"\n";

  }

  Vec<Float, 3> tangent_x, tangent_y;
  create_basis(reflect_dir, tangent_x, tangent_y);

  Vec<Float, 3> sample_dir = tangent_x * local_dir[0] +
                             tangent_y * local_dir[1] +
                             normal * local_dir[2];

  pdf = ((shine +2.f)/(2.f * pi())) * pow(cos(alpha),shine);
  return sample_dir;
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

void add(Array<Ray> &rays,
         Array<Vec<float32,3>> &input,
         Array<Vec<float32,4>> &output,
         Array<Vec<float32,3>> &att)
{
  Vec<float32,4> *output_ptr = output.get_device_ptr();
  const Vec<float32,3> *input_ptr = input.get_device_ptr_const();
  const Vec<float32,3> *att_ptr = att.get_device_ptr_const();
  const Ray *ray_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, input.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const int32 idx = ray_ptr[ii].m_pixel_id;
    Vec<float32,4> color = output_ptr[idx];
    // FINDME TODO
    Vec<float32,3> in = input_ptr[ii];
    if(idx == debug_ray) std::cout<<" ++++ adding "<<in<<" current atten "<<att_ptr[ii]<<"\n";

    color[0] += in[0];
    color[1] += in[1];
    color[2] += in[2];
    color[3] = 1.f;
    output_ptr[idx] = color;
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
    // TODO: transparency
    //color[3] = 1;
    color[0] = clamp(color[0],0.f,1.f);
    color[1] = clamp(color[1],0.f,1.f);
    color[2] = clamp(color[2],0.f,1.f);
    input_ptr[ii] = color;

  });
  DRAY_ERROR_CHECK();
}

struct DeviceSamples
{
  Vec<float32,4> *colors;
  Vec<float32,3> *normals;
  float32 *distances;
  int32 *hit_flags;
  int32 *material_ids;

  DeviceSamples() = delete;
  DeviceSamples(Samples &samples)
  {
    colors = samples.m_colors.get_device_ptr();
    normals = samples.m_normals.get_device_ptr();
    distances = samples.m_distances.get_device_ptr();
    hit_flags = samples.m_hit_flags.get_device_ptr();
    material_ids = samples.m_material_ids.get_device_ptr();
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
  samples.m_material_ids = gather(samples.m_material_ids, compact_idxs);
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
                     Array<Vec<float32,4>> &colors,
                     Samples &samples)
{
  DRAY_LOG_OPEN("update_samples");

  const RayHit *hit_ptr = hits.get_device_ptr_const ();
  const Fragment *frag_ptr = fragments.get_device_ptr_const ();

  DeviceSamples d_samples(samples);
  const Vec<float32,4> *color_ptr = colors.get_device_ptr_const();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];

    // could check the current distance
    // but we adjust the ray max do this shouldn't need to
    if (hit.m_hit_idx > -1)
    {
      d_samples.normals[ii] = frag.m_normal;
      d_samples.colors[ii]  = color_ptr[ii];
      d_samples.distances[ii] = hit.m_dist;
      d_samples.hit_flags[ii] = 1;
    }

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

void Samples::resize(int32 size)
{
  m_colors.resize(size);
  m_normals.resize(size);
  m_distances.resize(size);
  m_hit_flags.resize(size);
  m_material_ids.resize(size);
  array_memset_zero (m_hit_flags);
}

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

void TestRenderer::add_light(const QuadLight &light)
{
  m_quad_lights.push_back(light);
}

void TestRenderer::setup_lighting(Camera &camera)
{

  if(m_sphere_lights.size() + m_quad_lights.size() > 0)
  {
    m_lights.pack(m_sphere_lights, m_quad_lights);
  }
  else
  {
    SphereLight light = detail::test_default_light(camera, m_scene_bounds);
    std::vector<SphereLight> sphere_lights;
    sphere_lights.push_back(light);
    m_lights.pack(phere_lights, m_quad_lights);
  }

  if(m_sphere_lights.size() > 0)
  {
    m_sphere_array.resize(m_sphere_lights.size());
    SphereLight* light_ptr = m_sphere_array.get_host_ptr();
    for(int i = 0; i < m_sphere_lights.size(); ++i)
    {
      light_ptr[i] = m_sphere_lights[i];
    }
  }
  else if(m_quad_lights.size() == 0)
  {
    m_sphere_array.resize(1);
    SphereLight light = detail::test_default_light(camera, m_scene_bounds);
    SphereLight* light_ptr = m_sphere_array.get_host_ptr();
    light_ptr[0] = light;
  }

  if(m_quad_lights.size() > 0)
  {
    m_quad_array.resize(m_quad_lights.size());
    QuadLight* light_ptr = m_quad_array.get_host_ptr();
    for(int i = 0; i < m_quad_lights.size(); ++i)
    {
      light_ptr[i] = m_quad_lights[i];
    }
  }
}

void TestRenderer::use_lighting(bool use_it)
{
  m_use_lighting = use_it;
}

void TestRenderer::add(std::shared_ptr<Traceable> traceable, Material mat)
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
      //for(int h = 0; h < rays.size(); ++h)
      //{
      //  if(rays.get_value(h).m_pixel_id == debug_ray)
      //  {
      //    std::cout<<"nearest_hit "<<"("<<d<<") "<<hits.get_value(h).m_hit_idx
      //             <<" "<<hits.get_value(h).m_dist<<"\n";
      //  }
      //}
      Array<Vec<float32,4>> colors;
      m_traceables[i]->colors(rays, hits, fragments, colors);
      detail::update_samples(hits, fragments, colors, samples);
      ray_max(rays, hits);
    }
  }

  for(int h = 0; h < rays.size(); ++h)
  {
    if(rays.get_value(h).m_pixel_id == debug_ray)
    {
      std::cout<<"resulting hit dist " <<samples.m_distances.get_value(h)<<"\n";
      std::cout<<"              hit " <<samples.m_hit_flags.get_value(h)<<"\n";
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

  const int32 num_samples = m_num_samples;;
  for(int32 sample = 0; sample < num_samples; ++sample)
  {
    m_sample_count = sample;
    Array<Ray> rays;
    camera.create_rays_jitter (rays);

    int32 max_depth = 8;
    m_depth = 0;
    Array<Vec<float32,3>> attenuation;
    attenuation.resize(rays.size());
    Vec<float32,3> white = {{1.f,1.f,1.f}};
    array_memset_vec(attenuation, white);

    for(int32 depth = 0; depth < max_depth; ++depth)
    {
      m_depth = depth;

      Samples samples = nearest_hits(rays);
#ifdef RAY_DEBUGGING
      std::cout<<"Debugging "<<rays.size()<<"\n";
      for(int i = 0; i < rays.size(); ++i)
      {
        RayDebug debug;
        debug.ray = rays.get_value(i);
        debug.distance = samples.m_distances.get_value(i);
        debug.hit = samples.m_hit_flags.get_value(i);
        debug.depth = m_depth;
        debug.sample = m_sample_count;
        debug.shadow = 0;
        debug.red = attenuation.get_value(i)[0];
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
      intersect_lights(rays, samples, framebuffer, depth);

      // reduce to only the hits
      std::cout<<"Depth "<<depth<<" input rays "<<rays.size()<<"\n";
      detail::compact_hits(rays, samples, attenuation);
      std::cout<<"compact rays "<<rays.size()<<"\n";
      if(rays.size() == 0)
      {
        break;
      }

      // cast shadow rays
      Array<Vec<float32,3>> light_colors = direct_lighting(rays, samples);
      // add in the attenuation
      detail::multiply(light_colors, attenuation);
      // add the light contribution
      detail::add(rays, light_colors, framebuffer.colors(), attenuation);

      // bounce
      bounce(rays, samples);

      // attenuate the light
      detail::multiply(attenuation, samples.m_colors);

      if(depth >= 3)
      {
        russian_roulette(attenuation, samples, rays);
        if(rays.size() == 0)
        {
          break;
        }
      }

      for(int h = 0; h < rays.size(); ++h)
      {
        if(rays.get_value(h).m_pixel_id == debug_ray)
        {
          std::cout<<"Current Attenuation "<<attenuation.get_value(h)<<"\n";
        }
      }

    }
    //std::cout<<"Last ray "<<rays.get_value(0).m_pixel_id<<"\n";

  }

  detail::average(framebuffer.colors(), num_samples);


  std::cout<<"final color "<<framebuffer.colors().get_value(debug_ray)<<"\n";
  framebuffer.tone_map();
  std::cout<<"final color tone map"<<framebuffer.colors().get_value(debug_ray)<<"\n";

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

void TestRenderer::bounce(Array<Ray> &rays, Samples &samples)
{
  const int32 size = rays.size();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();
  detail::DeviceSamples d_samples(samples);
  Ray * ray_ptr = rays.get_device_ptr();

  const float32 eps = ray_eps;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    Ray ray = ray_ptr[ii];
    const float32 distance = d_samples.distances[ii];
    Vec<float32,3> hit_point = ray.m_orig + ray.m_dir * distance;
    Vec<float32,3> normal = d_samples.normals[ii];
    normal.normalize();

    if(dot(normal,-ray.m_dir) < 0)
    {
      normal = -normal;
    }

    bool debug = ray.m_pixel_id == debug_ray;
    Vec<float32,4> color = d_samples.colors[ii];
    Vec<uint32,2> rand_state = rand_ptr[ray.m_pixel_id];

    if(debug) std::cout<<"<Bounce>  in color "<<color<<"\n";


    float32 roll = randomf(rand_state);

    // choose between transmitting and reflecting
    if(roll < color[3] || true)
    {
      hit_point += eps * (-ray.m_dir);
      Vec<float32,2> rand;
      rand[0] = randomf(rand_state);
      rand[1] = randomf(rand_state);
      float32 test_val;
      Vec<float32,3> new_dir = detail::cosine_weighted_hemisphere (normal, rand, test_val);
      new_dir.normalize();
      float32 cos_theta = dot(new_dir, normal);
      if(debug) std::cout<<" test "<<test_val<<" cost "<<cos_theta<<"\n";
      color *= cos_theta;
      //float pdf = cos_theta / pi();
      //color /= pdf;
      if(debug) std::cout<<"   diffuse cos "<<cos_theta<<"\n";

      ray.m_dir = new_dir;
      ray.m_orig = hit_point;
    }
    //{
    //  hit_point += eps * (-ray.m_dir);
    //  Vec<float32,2> rand;
    //  rand[0] = randomf(rand_state);
    //  rand[1] = randomf(rand_state);
    //  float32 pdf;
    //  Vec<float32,3> new_dir = detail::phong_brdf(ray.m_dir, normal, 30.f, rand, pdf, debug);
    //  new_dir.normalize();
    //  color /= pdf;
    //  if(debug) std::cout<<" pdf "<<pdf<<"\n";
    //  if(debug) std::cout<<" dir "<<new_dir<<"\n";

    //  ray.m_dir = new_dir;
    //  ray.m_orig = hit_point;
    //}
    else
    {
      if(debug) std::cout<<" ===== trans \n";
      hit_point += eps * ray.m_dir;
      ray.m_orig = hit_point;
      // we want to attenuate this color with respect
      // to alpha
      color[0] = 1.f - color[0] * color[3];
      color[1] = 1.f - color[1] * color[3];
      color[2] = 1.f - color[2] * color[3];
      color[3] = 1.f;
    }

    if(debug) std::cout<<"   bounce attenuation "<<color<<"\n";


    ray.m_near = 0;
    ray.m_far = infinity<Float>();
    ray_ptr[ii] = ray;

    // update the random state
    rand_ptr[ray.m_pixel_id] = rand_state;
    d_samples.colors[ii] = color;

  });
}

Vec<float32,3>
sphere_sample(const Vec<float32,3> &center,
              const float32 &radius,
              const Vec<float32,3> &hit_point,
              const Vec<float32,2> &u, // random
              float32 &pdf,
              bool debug = false)
{

  Vec<float32, 3> light_dir = center - hit_point;

  float32 dc = light_dir.magnitude();

  if(dc < radius)
  {
    std::cout<<"Inside light\n";
  }

  float32 invDc = 1.f / dc;
  Vec<float32, 3> wc = (center - hit_point) * invDc;
  Vec<float32,3> wcX, wcY;
  detail::create_basis(wc,wcX,wcY);

  float32 sin_theta_max = radius * invDc;
  float32 sin_theta_max2 = sin_theta_max * sin_theta_max;
  float32 inv_sin_theta_max = 1 / sin_theta_max;
  float32 cos_theta_max = sqrt(max(0.f, 1.f - sin_theta_max2));

  float32 cos_theta = (cos_theta_max - 1.f) * u[0] + 1.f;
  float32 sin_theta2 = 1.f - cos_theta * cos_theta;

  float32 cos_alpha = sin_theta2 * inv_sin_theta_max + cos_theta *
    sqrt(max(0.f, 1.f - sin_theta2 * inv_sin_theta_max * inv_sin_theta_max));
  float32 sin_alpha = sqrt(max(0.f, 1.f - cos_alpha * cos_alpha));
  float32 phi = u[1] * 2.f * pi();


  // convert spherical coords, project to world
  // and get the point on the sphere. The coordinate
  // system was created to the sphere, so its reversed here
  Vec<float32,3> dir = sin_alpha * cos(phi) * (-wcX)
    + sin_alpha * sin(phi) * (-wcY) + cos_alpha * (-wc);
  Vec<float32,3> point = center + radius * dir;

  pdf = 1.f / (2.f * pi() * (1.f - cos_theta_max));
  if(debug) std::cout<<"cos theta max "<<cos_theta_max<<" pdf "<<pdf<<" \n";
  return point;
}

Array<Ray>
TestRenderer::create_shadow_rays(Array<Ray> &rays,
                                 Samples &samples,
                                 const SphereLight light,
                                 Array<float32> &inv_pdf)
{
  Array<Ray> shadow_rays;
  shadow_rays.resize(rays.size());
  inv_pdf.resize(rays.size());

  detail::DeviceSamples d_samples(samples);

  const Ray *ray_ptr = rays.get_device_ptr_const ();
  //const Vec<float32,3> *normals_ptr = normals.get_device_ptr_const ();
  //const float32 *distances_ptr = distances.get_device_ptr_const ();

  float32 *inv_pdf_ptr = inv_pdf.get_device_ptr();
  Ray *shadow_ptr = shadow_rays.get_device_ptr();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();

  const float32 eps = ray_eps;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray &ray = ray_ptr[ii];
    bool debug = ray.m_pixel_id == debug_ray;
    const float32 distance = d_samples.distances[ii];
    Vec<float32,3> hit_point = ray.m_orig + ray.m_dir * distance;
    // back it away a bit
    hit_point += eps * (-ray.m_dir);
    if(debug) std::cout<<"hit pos "<<hit_point<<"\n";


    const float32 radius = light.m_radius;

    Vec<uint32,2> rand_state = rand_ptr[ray.m_pixel_id];
    Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);
    // update the random state
    rand_ptr[ray.m_pixel_id] = rand_state;

    float32 pdf;
    Vec<float32,3> sample_point = sphere_sample(light.m_pos,
                                                light.m_radius,
                                                hit_point,
                                                rand,
                                                pdf,
                                                debug);

    Vec<float32,3> sample_dir = sample_point - hit_point;
    float32 sample_distance = sample_dir.magnitude();
    sample_dir.normalize();

    if(debug)
    {
      std::cout<<" light sample dir "<<sample_dir<<"\n";
      std::cout<<"   hit "<<hit_point<<"\n";
      std::cout<<"   distance "<<sample_distance<<"\n";
      std::cout<<"   rand "<<rand<<"\n";
    }

    Vec<float32,3> normal = d_samples.normals[ii];

    if(dot(normal,-ray.m_dir) < 0)
    {
      normal = -normal;
    }

    normal.normalize();

    float32 dot_ns = dot(normal,sample_dir);
    bool valid = dot_ns > 0;

    float32 inv_pdf = 1.0f / pdf;

    if(!valid)
    {
      // TODO: just never cast this ray
      inv_pdf = 0.f;
    }

    float32 alpha = d_samples.colors[ii][3];
    inv_pdf_ptr[ii] = inv_pdf * dot_ns * alpha;

    if(debug)
    {
      std::cout<<"  shadow weight "<<inv_pdf_ptr[ii]<<"\n";
      std::cout<<"     color "<<d_samples.colors[ii]<<"\n";
      std::cout<<"     dot "<<dot_ns<<"\n";
      std::cout<<"     invpdf "<<inv_pdf<<"\n";
      std::cout<<"     alpha  "<<alpha<<"\n";

    }

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
TestRenderer::shade_lights(const Vec<float32,3> light_color,
                           Array<Ray> &rays,
                           Array<Ray> &shadow_rays,
                           Array<Vec<float32,3>> &normals,
                           Array<int32> &hit_flags,
                           Array<float32> &inv_pdf,
                           Array<Vec<float32,3>> &colors)
{
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const Ray *shadow_ptr = shadow_rays.get_device_ptr_const();
  const int32 *hit_flag_ptr = hit_flags.get_device_ptr_const();
  const Vec<float32,3> *normal_ptr = normals.get_device_ptr_const();
  const float32 *inv_pdf_ptr = inv_pdf.get_device_ptr_const();
  Vec<float32,3> *color_ptr = colors.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    // TODO: get rid of ray
    const Ray &ray = ray_ptr[ii];
    const Ray &shadow_ray = shadow_ptr[ii];
    Vec<float32,3> normal = normal_ptr[ii];
    normal.normalize();

    if(dot(normal,shadow_ray.m_dir) < 0)
    {
      normal = -normal;
    }

    Vec<float32,3> color = {{0.f, 0.f, 0.f}};

    if(hit_flag_ptr[ii] == 0)
    {
      color = light_color * max(0.f, dot(normal, shadow_ray.m_dir));
    }

    float32 inv_pdf = inv_pdf_ptr[ii];

    color_ptr[ii] += color * inv_pdf;

    if(shadow_ray.m_pixel_id == debug_ray)
    {
      std::cout<<" +++--- light color "<<color_ptr[ii]<<"\n";
      std::cout<<" +++--- hit flag "<<hit_flag_ptr[ii]<<"\n";
    }

  });
}

Array<Vec<float32,3>>
TestRenderer::direct_lighting(Array<Ray> &rays,
                              Samples &samples)
{
  Array<Vec<float32,3>> contributions;
  contributions.resize(rays.size());
  Vec<float32,3> black = {{0.f, 0.f, 0.f}};
  array_memset_vec (contributions, black);

  for(int l = 0; l < m_sphere_array.size(); ++l)
  {
    SphereLight light = m_sphere_array.get_value(l);

    Array<float32> inv_pdf;
    Array<Ray> shadow_rays = create_shadow_rays(rays,
                                                samples,
                                                light,
                                                inv_pdf);

    Array<int32> hit_flags = any_hit(shadow_rays);

    shade_lights(light.m_intensity,
                 rays,
                 shadow_rays,
                 samples.m_normals,
                 hit_flags,
                 inv_pdf,
                 contributions);
  }

  // multiply the light contributions by the input color
  detail::multiply(contributions, samples.m_colors);
  return contributions;

}

void TestRenderer::intersect_lights(Array<Ray> &rays,
                                    Samples &samples,
                                    Framebuffer &fb,
                                    int32 depth)
{
  detail::DeviceSamples d_samples(samples);
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const SphereLight *lights_ptr = m_sphere_array.get_device_ptr_const();
  const int32 num_lights = m_sphere_array.size();
  Vec<float32,4> *color_ptr = fb.colors().get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray ray = ray_ptr[ii];
    int32 light_id = -1;
    float32 nearest_dist = d_samples.distances[ii];
    int32 hit = d_samples.hit_flags[ii];
    if(hit != 1)
    {
      nearest_dist = infinity32();
    }

    for(int32 i = 0; i < num_lights; ++i)
    {
      const SphereLight light = lights_ptr[i];
      float32 dist = detail::intersect_sphere(light.m_pos,
                                              light.m_radius,
                                              ray.m_orig,
                                              ray.m_dir);
      if(dist < nearest_dist && dist < ray.m_far)
      {
        light_id = i;
        nearest_dist = dist;
      }
    }

    if(light_id != -1)
    {
      // Kill this ray
      d_samples.hit_flags[ii] = 0;
      if(depth == 0)
      {
        Vec<float32,3> l_color =  lights_ptr[light_id].m_intensity;
        color_ptr[ray.m_pixel_id][0] += l_color[0];
        color_ptr[ray.m_pixel_id][1] += l_color[1];
        color_ptr[ray.m_pixel_id][2] += l_color[2];
        color_ptr[ray.m_pixel_id][3] = 1.f;
      }
    }

  });
  DRAY_ERROR_CHECK();
}

void TestRenderer::russian_roulette(Array<Vec<float32,3>> &attenuation,
                                    Samples &samples,
                                    Array<Ray> &rays)
{
  Array<int32> keep_flags;
  keep_flags.resize(rays.size());
  int32 *keep_ptr = keep_flags.get_device_ptr();
  Vec<uint32,2> *rand_ptr = m_rand_state.get_device_ptr();

  Vec<float32,3> *atten_ptr = attenuation.get_device_ptr();
  const Ray *ray_ptr = rays.get_device_ptr_const();


  RAJA::forall<for_policy> (RAJA::RangeSegment (0, rays.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    int32 keep = 1;
    Vec<float32,3> att = atten_ptr[ii];
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
      //att *= 1.f/(1. - q);
      if(debug && keep == 1)
      {
        std::cout<<"Russian attenuation correction "<<(1.f/(1. - q))
                 <<" roll "<<roll<<" q "<<q<<" max att "<<max_att<<"\n";
      }
    }

    if(debug)
    {
      std::cout<<"Russian keep "<<keep<<"\n";
    }
    atten_ptr[ii] = att;
    keep_ptr[ii] = keep;
    rand_ptr[pixel_id] = rand_state;
  });

  DRAY_ERROR_CHECK();

  int32 before_size = rays.size();

  Array<int32> compact_idxs = index_flags(keep_flags);

  rays = gather(rays, compact_idxs);
  samples.m_colors = gather(samples.m_colors, compact_idxs);
  samples.m_normals = gather(samples.m_normals, compact_idxs);
  samples.m_distances = gather(samples.m_distances, compact_idxs);
  samples.m_hit_flags = gather(samples.m_hit_flags, compact_idxs);
  samples.m_material_ids = gather(samples.m_material_ids, compact_idxs);
  attenuation = gather(attenuation, compact_idxs);

  std::cout<<" Russian culled "<<before_size - rays.size()<<"\n";
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

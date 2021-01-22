// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRENDERER_HPP
#define DRAY_TRENDERER_HPP

#include <dray/rendering/camera.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/sphere_light.hpp>
#include <dray/rendering/traceable.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/rendering/path_data.hpp>
#include <dray/rendering/env_map.hpp>
#include <dray/random.hpp>

#include <memory>
#include <vector>

namespace dray
{

struct RayDebug
{
  int depth = 0;
  int sample = 0;
  int shadow = 0;
  Ray ray;
  bool hit;
  float distance;
  float red;
};

class TestRenderer
{
protected:
  std::map<int32,std::vector<RayDebug>> debug_geom;
  std::vector<std::shared_ptr<Traceable>> m_traceables;
  std::vector<Material> m_materials;
  Array<Material> m_materials_array;
  std::shared_ptr<Volume> m_volume;

  std::vector<SphereLight> m_sphere_lights;
  std::vector<TriangleLight> m_tri_lights;

  LightContainer m_lights;

  bool m_use_lighting;
  bool m_screen_annotations;
  Array<Vec<uint32,2>> m_rand_state;
  AABB<3> m_scene_bounds;
  int32 m_depth;
  int32 m_sample_count;
  int32 m_num_samples;
  EnvMap m_env_map;

  Array<Sample> nearest_hits(Array<Ray> &rays);
  Array<int32> any_hit(Array<Ray> &rays);


public:
  TestRenderer();
  void clear();
  void clear_lights();
  void add(std::shared_ptr<Traceable> traceable, Material mat);
  void volume(std::shared_ptr<Volume> volume);

  void add_light(const SphereLight &light);
  void add_light(const TriangleLight &light);
  void setup_lighting(Camera &camera);
  void setup_materials();
  void load_env_map(const std::string filename);

  void use_lighting(bool use_it);
  Framebuffer render(Camera &camera);
  void screen_annotations(bool on);

  Array<Vec<float32,3>> direct_lighting(Array<Ray> &rays,
                                        Array<Sample> &samples);

  // check
  void intersect_lights(Array<Ray> &rays,
                        Array<Sample> &samples,
                        Array<RayData> &data,
                        Framebuffer &fb,
                        int32 depth);

  void bounce(Array<Ray> &rays,
              Array<RayData> &ray_data,
              Array<Sample> &samples,
              Array<Material> &materials);

  Array<Ray> create_shadow_rays(Array<Ray> &rays,
                                Array<Sample> &samples,
                                Array<Vec<float32,3>> &light_color,
                                Array<Material> &materials);

  void shade_lights(Array<int32> &hit_flags,
                    Array<Vec<float32,3>> &light_colors,
                    Array<Vec<float32,3>> &colors);

  void russian_roulette(Array<RayData> &data,
                        Array<Sample> &samples,
                        Array<Ray> &rays);

  void write_debug(Framebuffer &fb);

  void samples(int32 num_samples);

};


} // namespace dray
#endif

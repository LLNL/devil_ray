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
#include <dray/random.hpp>

#include <memory>
#include <vector>

namespace dray
{

struct Samples
{
  Array<Vec<float32,4>> m_colors;
  Array<Vec<float32,3>> m_normals;
  Array<float32> m_distances;
  Array<int32> m_hit_flags;
  Array<int32> m_material_ids;
  void resize(int32 size);
};

struct Material
{
  Vec3f m_specular = {{0.9f, 0.9f, 0.9f}};;
  Vec3f m_emmisive = {{0.f, 0.f, 0.f}};;
  float32 m_reflectiviy = 0.25f;
  float32 m_transmittance = 0.50f;
  //float32 m_ior = -1.f;
};

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
  std::vector<int32> m_material_ids;
  std::shared_ptr<Volume> m_volume;
  std::vector<SphereLight> m_lights;
  bool m_use_lighting;
  bool m_screen_annotations;
  Array<Vec<uint32,2>> m_rand_state;
  AABB<3> m_scene_bounds;
  int32 m_depth;
  int32 m_num_samples;

  Samples nearest_hits(Array<Ray> &rays);
  Array<int32> any_hit(Array<Ray> &rays);

public:
  TestRenderer();
  void clear();
  void clear_lights();
  void add(std::shared_ptr<Traceable> traceable, Material mat);
  void volume(std::shared_ptr<Volume> volume);
  void add_light(const SphereLight &light);
  void use_lighting(bool use_it);
  Framebuffer render(Camera &camera);
  void screen_annotations(bool on);

  Array<Vec<float32,3>> direct_lighting(Array<SphereLight> &lights,
                                        Array<Ray> &rays,
                                        Samples &samples);

  // check
  void intersect_lights(Array<SphereLight> &lights,
                        Array<Ray> &rays,
                        Samples &samples);

  void bounce(Array<Ray> &rays, Samples &samples);

  Array<Ray> create_shadow_rays(Array<Ray> &rays,
                                Array<float32> &distances,
                                Array<Vec<float32,3>> &normals,
                                const SphereLight light,
                                Array<float32> &inv_pdf);

  void shade_lights(const Vec<float32,3> light_color,
                    Array<Ray> &rays,
                    Array<Ray> &shadow_rays,
                    Array<Vec<float32,3>> &normals,
                    Array<int32> &hit_flags,
                    Array<float32> &inv_pdf,
                    Array<Vec<float32,3>> &colors);

  void write_debug();

  void samples(int32 num_samples);

};


} // namespace dray
#endif

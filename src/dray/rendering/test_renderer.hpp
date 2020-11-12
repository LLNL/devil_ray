// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRENDERER_HPP
#define DRAY_TRENDERER_HPP

#include <dray/rendering/camera.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/point_light.hpp>
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

  void resize(int32 size);
};

struct RayDebug
{
  int depth;
  int sample;
  float start;
  float end;
}

class TestRenderer
{
protected:
  std::map<int32,std::vector<std::pair<float,float>>> debug_geom;
  std::vector<std::shared_ptr<Traceable>> m_traceables;
  std::shared_ptr<Volume> m_volume;
  std::vector<PointLight> m_lights;
  bool m_use_lighting;
  bool m_screen_annotations;
  Array<Vec<uint32,2>> m_rand_state;

  Samples nearest_hits(Array<Ray> &rays);
  Array<int32> any_hit(Array<Ray> &rays);

public:
  TestRenderer();
  void clear();
  void clear_lights();
  void add(std::shared_ptr<Traceable> traceable);
  void volume(std::shared_ptr<Volume> volume);
  void add_light(const PointLight &light);
  void use_lighting(bool use_it);
  Framebuffer render(Camera &camera);
  void screen_annotations(bool on);

  Array<Vec<float32,3>> direct_lighting(Array<PointLight> &lights,
                                        Array<Ray> &rays,
                                        Samples &samples);

  void bounce(Array<Ray> &rays, Samples &samples);

  Array<Ray> create_shadow_rays(Array<Ray> &rays,
                                Array<float32> &distances,
                                const Vec<float32,4> sphere);

  void shade_lights(const Vec<float32,3> light_color,
                    Array<Ray> &rays,
                    Array<Ray> &shadow_rays,
                    Array<Vec<float32,3>> &normals,
                    Array<int32> &hit_flags,
                    Array<Vec<float32,3>> &colors);

  void write_debug();
};


} // namespace dray
#endif

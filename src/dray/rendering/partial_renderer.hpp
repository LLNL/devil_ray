// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_PARTIAL_RENDERER_HPP
#define DRAY_PARTIAL_RENDERER_HPP

#include <dray/data_set.hpp>
#include <dray/color_map.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/rendering/volume_partial.hpp>
#include <dray/rendering/point_light.hpp>

namespace dray
{

class PartialRenderer
{
protected:
  int32 m_samples;
  ColorMap m_color_map;
  DataSet m_data_set;
  DataSet m_boundary;
  std::string m_field;
public:
  PartialRenderer() = delete;
  PartialRenderer(DataSet &data_set);
  ~PartialRenderer();
  //Array<RayHit> nearest_hit(Array<Ray> &rays);
  // volume rendering is a bit different
  void integrate(Array<Ray> &rays, Array<PointLight> &lights);

  /// set the input data set
  void input(DataSet &data_set);

  /// set the number of samples
  void samples(int32 num_samples);

  void field(const std::string field);
};


} // namespace dray
#endif

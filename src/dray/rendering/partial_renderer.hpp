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
  std::string m_field;
  AABB<3> m_bounds;

public:
  PartialRenderer() = delete;
  PartialRenderer(DataSet &data_set);
  ~PartialRenderer();

  Array<VolumePartial> integrate(Array<Ray> &rays, Array<PointLight> &lights);

  void save(const std::string name,
            Array<VolumePartial> partials,
            const int32 width,
            const int32 height);

  /// set the input data set
  void input(DataSet &data_set);

  /// set the number of samples based on the bounds. If no
  //bounds is passed in, we use the current data set bounds
  void samples(int32 num_samples, AABB<3> bounds = AABB<3>());

  void field(const std::string field);

  ColorMap& color_map();
};


} // namespace dray
#endif

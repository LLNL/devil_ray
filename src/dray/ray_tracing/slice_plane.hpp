// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SLICE_PLANE_HPP
#define DRAY_SLICE_PLANE_HPP

#include <dray/ray_tracing/traceable.hpp>

namespace dray
{
namespace ray_tracing
{

class SlicePlane : public Traceable
{
  Vec<float32,3> m_point;
  Vec<float32,3> m_normal;
public:
  SlicePlane() = delete;
  SlicePlane(DataSet &data_set);

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays);

  template<class MeshElement>
  Array<RayHit> execute(Mesh<MeshElement> &mesh, Array<Ray> &rays);

  void set_point(const Vec<float32,3> &point);
  void set_normal(const Vec<float32,3> &normal);

};

}};//namespace dray::ray_tracing

#endif//DRAY_VOLUME_INTEGRATOR_HPP

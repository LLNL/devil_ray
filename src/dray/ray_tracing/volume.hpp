// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_VOLUME_HPP
#define DRAY_VOLUME_HPP

#include <dray/ray_tracing/traceable.hpp>

namespace dray
{
namespace ray_tracing
{

class Volume : public Traceable
{
protected:
  int32 m_samples;

public:
  Volume() = delete;
  Volume(DataSet &data_set);
  virtual ~Volume();
  virtual Array<RayHit> nearest_hit(Array<Ray> &rays);
  // volume rendering is a bit different
  void integrate(Array<Ray> &rays, Framebuffer &fb, Array<PointLight> &lights);

  template<typename MeshElement, typename FieldElement>
  void integrate(Mesh<MeshElement> &mesh,
                 Field<FieldElement> &field,
                 Array<Ray> &rays,
                 Framebuffer &fb,
                 Array<PointLight> &lights);
  /// set the input data set
  void input(DataSet &data_set);
  /// sets the field for that generates fragments for shading
  virtual bool is_volume() const override;
};


}} // namespace dray::ray_tracing
#endif

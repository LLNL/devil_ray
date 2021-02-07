// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POINTS_HPP
#define DRAY_POINTS_HPP

#include<dray/rendering/traceable.hpp>

namespace dray
{

class Points : public Traceable
{
protected:
  // this will change with the addition of fields
  Vec<float32,4> m_color;

public:
  Points() = delete;
  Points(Collection &collection);
  virtual ~Points();

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) override;

  virtual void shade(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     const Array<PointLight> &lights,
                     Framebuffer &framebuffer) override;

  virtual void shade(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     Framebuffer &framebuffer) override;

  virtual void colors(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     Array<Vec<float32,4>> &colors) override;

  virtual Array<Fragment> fragments(Array<RayHit> &hits) override;

  void constant_color(const Vec<float32,4> &color);
};

};//namespace dray

#endif //DRAY_SURFACE_HPP

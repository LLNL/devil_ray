// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SURFACE_HPP
#define DRAY_SURFACE_HPP

#include<dray/ray_tracing/traceable.hpp>

namespace dray
{
namespace ray_tracing
{

class Surface : public Traceable
{
public:
  Surface() = delete;
  Surface(DataSet &dataset);

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) override;

  template<typename MeshElem>
  Array<RayHit> execute(Mesh<MeshElem> &mesh, Array<Ray> &rays);
};

}};//namespace dray::ray_tracing

#endif //DRAY_SURFACE_HPP

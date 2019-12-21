// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRACEABLE_HPP
#define DRAY_TRACEABLE_HPP

#include <dray/array.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/data_set.hpp>

namespace dray
{
namespace ray_tracing
{

class Traceable
{
protected:
  DataSet m_data_set;
public:
  Traceable() = delete;
  Traceable(DataSet &data_set);
  virtual ~Traceable();

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) = 0;

  void input(DataSet &data_set);
};


}} // namespace dray::ray_tracing
#endif

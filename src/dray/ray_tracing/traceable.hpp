// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRACEABLE_HPP
#define DRAY_TRACEABLE_HPP

#include <dray/array.hpp>
#include <dray/fragment.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/data_set.hpp>

namespace dray
{
namespace ray_tracing
{
/**
 * \class Traceable
 * \brief Encapsulates a traceable object
 *
 * Defines the interface for traceable objects
 */

class Traceable
{
protected:
  DataSet m_data_set;
  std::string m_field_name;
public:
  Traceable() = delete;
  Traceable(DataSet &data_set);
  virtual ~Traceable();
  /// returns the nearests hit along a batch of rays
  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) = 0;
  /// returns the fragments for a batch of hits
  virtual Array<Fragment> fragments(Array<RayHit> &hits);

  /// set the input data set
  void input(DataSet &data_set);
  /// sets the field for that generates fragments for shading
  void field(const std::string &field_name);
};


}} // namespace dray::ray_tracing
#endif

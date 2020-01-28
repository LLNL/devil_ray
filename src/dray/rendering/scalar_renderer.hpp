// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SCALER_RENDERER_HPP
#define DRAY_SCALER_RENDERER_HPP

#include <dray/rendering/camera.hpp>
#include <dray/rendering/scalar_buffer.hpp>
#include <dray/rendering/traceable.hpp>

#include <memory>
#include <vector>

namespace dray
{

class ScalarRenderer
{
protected:
  std::shared_ptr<Traceable> m_traceable;
  void ray_max(Array<Ray> &rays, const Array<RayHit> &hits) const;
public:
  void set(std::shared_ptr<Traceable> traceable);
  ScalarBuffer render(Camera &camera);
};


} // namespace dray
#endif

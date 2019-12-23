// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RENDERER_HPP
#define DRAY_RENDERER_HPP

#include <dray/camera.hpp>
#include <dray/framebuffer.hpp>
#include <dray/ray_tracing/traceable.hpp>

#include <memory>
#include <vector>

namespace dray
{
namespace ray_tracing
{

class Renderer
{
protected:
  std::vector<std::shared_ptr<Traceable>> m_traceables;
public:
  void clear();
  void add(std::shared_ptr<Traceable> &traceable);
  Framebuffer render(Camera &camera);
};


}} // namespace dray::ray_tracing
#endif

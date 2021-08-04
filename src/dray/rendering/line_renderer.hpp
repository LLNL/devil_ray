// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LINE_RENDERER_HPP
#define DRAY_LINE_RENDERER_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/rendering/framebuffer.hpp>

namespace dray
{

class LineRenderer
{
public:
  void render(Framebuffer &fb, Array<Vec<float32,3>> starts, Array<Vec<float32,3>> ends);
};

} // namespace dray

#endif

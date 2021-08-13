// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LINE_RENDERER_HPP
#define DRAY_LINE_RENDERER_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/matrix.hpp>
#include <dray/rendering/framebuffer.hpp>

namespace dray
{

class LineRenderer
{
public:
  void render(
  	Framebuffer &fb, 
  	Matrix<float32, 4, 4> transform,
  	Array<Vec<float32,3>> starts, 
  	Array<Vec<float32,3>> ends);
  void justinrender(
  	Framebuffer &fb, 
  	Matrix<float32, 4, 4> transform,
  	Array<Vec<float32,3>> starts, 
  	Array<Vec<float32,3>> ends);
};

} // namespace dray

#endif

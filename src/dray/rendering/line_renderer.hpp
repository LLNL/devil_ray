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
#include <dray/rendering/renderer.hpp>
#include <dray/transform_3d.hpp>

namespace dray
{

class LineRenderer
{
public:
  void render_triad(
    Framebuffer &fb,
    Vec<int32, 2> pos,
    float32 distance,
    Camera &camera);
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

void crop_line_to_bounds(
	Vec<int32, 2> &p1, 
	Vec<int32, 2> &p2, 
	int32 width, 
	int32 height);

} // namespace dray

#endif

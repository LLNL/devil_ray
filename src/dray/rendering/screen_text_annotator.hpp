// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SS_TEXT_ANNOTATOR_HPP
#define DRAY_SS_TEXT_ANNOTATOR_HPP

#include <vector>

#include <dray/aabb.hpp>
#include <dray/rendering/framebuffer.hpp>

namespace dray
{

class ScreenTextAnnotator
{
protected:
  std::vector<std::vector<AABB<2>>> m_pixel_boxs;
  std::vector<std::vector<AABB<2>>> m_texture_boxs;
  std::string m_font_name;
public:
  ScreenTextAnnotator();
  void clear();
  void add_text(const std::string text,
                const Vec<float32,2> &pos,
                const float32 size);
  void render(Framebuffer &fb);
};

} // namespace dray
#endif

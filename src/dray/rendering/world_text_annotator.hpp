// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_WORLD_TEXT_ANNOTATOR_HPP
#define DRAY_WORLD_TEXT_ANNOTATOR_HPP

#include <vector>

#include <dray/aabb.hpp>
#include <dray/rendering/framebuffer.hpp>

namespace dray
{

class WorldTextAnnotator
{
protected:
  // std::vector<std::vector<AABB<2>>> m_world_boxs;
  std::vector<std::vector<AABB<2>>> m_pixel_boxs;
  std::vector<std::vector<AABB<2>>> m_texture_boxs;
  std::string m_font_name;
  std::vector<float32> m_depths;
public:
  WorldTextAnnotator();
  void clear();
  // void add_text(const std::string text,
  //               const Vec<float32,3> &world_pos,
  //               const float32 world_size);
  void add_text(const std::string text,
                const Vec<float32,2> &pos,
                const float32 size,
                const float32 depth);
  void render(Framebuffer &fb);
};

} // namespace dray
#endif

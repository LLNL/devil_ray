// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/world_text_annotator.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/rendering/font_factory.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/rendering/rasterbuffer.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>

#include<cmath>

namespace dray
{


WorldTextAnnotator::WorldTextAnnotator()
  : m_font_name("OpenSans-Regular")
{

}

void
WorldTextAnnotator::clear()
{
  //m_pixel_boxs.clear();
  //m_texture_boxs.clear();
}

void
WorldTextAnnotator::add_text(const std::string text,
                             const Vec<float32,3> &world_pos,
                             const float32 world_size)
{
  Font *font = FontFactory::font(m_font_name);

  // we are going to treat the font size as size in world space
  font->font_size(world_size);
  std::vector<AABB<2>> world_boxs;
  std::vector<AABB<2>> texture_boxs;
  Vec<float32,2> dims = {{world_pos[0], world_pos[1]}};
  AABB<2> tot = font->font_boxs(text, world_pos, world_boxs, texture_boxs);
  m_world_boxs.push_back(std::move(world_boxs));
  m_texture_boxs.push_back(std::move(texture_boxs));
  for(int i = 0; i < world_boxs.size(); ++i)
  {
    m_depths.push_back(world_pos[2]);
  }
}


void TextAnnotator::render(Framebuffer &fb)
{
  Array<float32> verts;
  Array<int32> indices;

  const int32 total_tris = m_world_boxs.size() * 2;
  const int32 total_verts = m_world_boxs.size() * 4;
  verts.resize(total_verts);

}

} // namespace dray

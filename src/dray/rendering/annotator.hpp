// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ANNOTATOR_HPP
#define DRAY_ANNOTATOR_HPP

#include <vector>

#include <dray/aabb.hpp>
#include <dray/color_map.hpp>
#include <dray/rendering/framebuffer.hpp>

namespace dray
{

class Annotator
{
protected:
  std::vector<AABB<2>> m_color_bar_pos;
public:
  Annotator();

  void screen_annotations(Framebuffer &fb,
                          const std::vector<std::string> &field_names,
                          std::vector<ColorMap> &color_maps);

};

} // namespace dray
#endif

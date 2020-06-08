// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/dray.hpp>
#include <dray/rendering/font.hpp>
#include <dray/rendering/text_annotator.hpp>

TEST (dray_smoke, dray_font)
{
  dray::TextAnnotator annot;
  dray::Vec<dray::float32,2> pos({{9.f,512.f}});
  annot.add_text("bananas", pos, 100.f);
  dray::Framebuffer fb;
  fb.clear();
  annot.render(fb);
  fb.save("annots");
  //dray::Font doit("MonospaceTypewriter");
  ////dray::Font doit("impact");

  //doit.font_size(29.f);
  //doit.write_test("bananas");

  //doit.doit();
}

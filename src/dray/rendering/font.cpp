// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/array.hpp>
#include <dray/error.hpp>
#include <dray/rendering/font.hpp>
#include <dray/utils/png_decoder.hpp>
#include <dray/utils/png_encoder.hpp>

#include<vector>
#include<cmath>

namespace dray
{

Font::Font()
  : m_valid(false),
    m_font_size(16.f)
{
}

Font::Font(const std::string font_file)
  : m_valid(false),
    m_font_size(16.f)
{
  load(font_file);
}

bool Font::valid() const
{
  return m_valid;
}

void Font::font_size(const float size)
{
  m_font_size = size;
}

float Font::font_size() const
{
  return m_font_size;
}

void Font::load(const std::string font_file)
{

  m_metadata.reset();
  m_valid = false;

  try
  {
    m_metadata.load(font_file + ".yaml", "yaml");
    const std::string image_name = font_file + ".png";
    const int width = m_metadata["bitmap_width"].to_int32();
    const int height = m_metadata["bitmap_height"].to_int32();

    m_texture.resize(width * height);

    PNGDecoder decoder;
    uint8 *buffer = nullptr;
    int b_width, b_height;
    decoder.decode(buffer, b_width, b_height, image_name);
    if(b_width != width || b_height != height)
    {
      std::cout<<"Mismatched image dims\n";
    }

    float32 *image_ptr = m_texture.get_host_ptr();
    for(int i = 0; i < width * height; ++i)
    {
      image_ptr[i] = float32(buffer[i*4+0]) / 255.f;
    }
    free(buffer);
    m_valid = true;
  }
  catch(const conduit::Error &e)
  {
    DRAY_ERROR("Failed to load font\n");
  }
  catch (...)
  {
    DRAY_ERROR("Unknown failure: load font '"<<font_file<<"'");
  }
}

int32 Font::texture_width() const
{
  return m_metadata["bitmap_width"].to_int32();
}

int32 Font::texture_height() const
{
  return m_metadata["bitmap_height"].to_int32();
}

void Font::write_test(const std::string text)
{
  const int t_width= 300;
  const int t_height = 100;

  std::vector<AABB<2>> pboxs;
  std::vector<AABB<2>> tboxs;
  Vec<float32,2> pos;
  pos[0] = 10.f;
  pos[1] = 10.f;

  AABB<2> tot = font_boxs(text, pos, pboxs, tboxs);
  std::cout<<"Total box "<<tot<<"\n";

  Array<Vec<float32,4>> test_image;
  test_image.resize(t_height * t_width);
  Vec<float32,4> *test_ptr = test_image.get_host_ptr();
  for(int i = 0; i < t_height * t_width; ++i)
  {
    test_ptr[i][0] = 0.f;
    test_ptr[i][1] = 0.f;
    test_ptr[i][2] = 0.f;
    test_ptr[i][3] = 1.f;
  }

  const int size = pboxs.size();
  float32 *image_ptr = m_texture.get_host_ptr();

    std::vector<Vec<float32,4>> colors;
    colors.push_back({{1.f, 0.f, 0.0f, 1.f}});
    colors.push_back({{0.f, 1.f, 0.0f, 1.f}});
    colors.push_back({{1.f, 1.f, 0.0f, 1.f}});
    colors.push_back({{0.f, 0.f, 1.0f, 1.f}});

    int count = 0;

  for(int i = 0; i < size; ++i)
  {
    const AABB<2> &pbox = pboxs[i];
    const AABB<2> &tbox = tboxs[i];
    float32 x_start = std::ceil(pbox.m_ranges[0].min());
    float32 x_end = std::floor(pbox.m_ranges[0].max());
    float32 y_start = std::ceil(pbox.m_ranges[1].min());
    float32 y_end = std::floor(pbox.m_ranges[1].max());

    float s0 = tbox.m_ranges[0].min();
    float s1 = tbox.m_ranges[0].max();
    float t0 = tbox.m_ranges[1].min();
    float t1 = tbox.m_ranges[1].max();
    std::cout<<"tbox "<<tbox<<"\n";
    std::cout<<"xstart : "<<x_start<<" y_start "<<y_start
             <<" x_end "<<x_end<<" y_end "<<y_end<<"\n";

    float width = 1024;
    float height = 1024;

    for(float yc = y_start; yc <= y_end; yc += 1.f)
    {
      // its upside down in texture coords so invert
      float tx = (yc - (pbox.m_ranges[1].min())) / pbox.m_ranges[1].length();
      //std::cout<<"tx "<<tx<<"\n";
      float t = lerp(t1, t0, tx) * height;
      for(float xc = x_start; xc <= x_end; xc += 1.f)
      {
        float sx = (xc - pbox.m_ranges[0].min()) / pbox.m_ranges[0].length();
        //std::cout<<"sx "<<sx<<"\n";
        float s = lerp(s0, s1, sx) * width;
        //std::cout<<"s "<<s<<" t "<<t<<"\n";
        int text_pixel_x = (int) s;
        int text_pixel_y = (int) t;
        int text_idx = text_pixel_y * width + text_pixel_x;

        float32 char_color = image_ptr[text_idx];
        //std::cout<<char_color<<"\n";

        //if(char_color[0] != 0.f)
        {
          int pixel_x = int(xc);
          int pixel_y = int(yc);
          int pixel_idx = pixel_y * t_width + pixel_x;

          Vec<float32,4> pixel_color;
          pixel_color[0] = char_color;
          pixel_color[1] = char_color;
          pixel_color[2] = char_color;
          pixel_color[3] = 1.f;
          //pixel_color = colors[count];
          test_ptr[pixel_idx] = pixel_color;
        }

      }
    }
    count++;
  }

  PNGEncoder encoder;
  encoder.encode((float*)test_ptr, t_width, t_height);
  encoder.save("text.png");
}

AABB<2> Font::font_boxs(const std::string text,
                        const Vec<float32,2> &pos,
                        std::vector<AABB<2>> &pixel_boxs,
                        std::vector<AABB<2>> &texture_boxs)
{

  if(!m_valid)
  {
    DRAY_ERROR("Font invalid\n");
  }
  Vec<float32,2> pen = pos;
  AABB<2> tot_aabb;
  std::string prev_char;
  for (auto it = text.begin(); it != text.end(); ++it)
  {
    std::cout<<"Pen "<<pen<<"\n";
    std::string character = string(1,*it);
    std::cout<<character<<"\n";
    if(!m_metadata.has_path("glyph_data/"+character))
    {
      DRAY_ERROR("Font: no character "<<*it);
    }
    const conduit::Node &glyph = m_metadata["glyph_data/"+string(1,*it)];
    glyph.print();

    float kerning = 0.f;
    if(it != text.begin() && glyph.has_path("kernings/"+prev_char))
    {
      kerning = glyph["kernings/"+prev_char].to_float32();
    }
    pen[0] += kerning * m_font_size;
    std::cout<<"kerning "<<kerning * m_font_size<<"\n";

    float32 width     = glyph["bbox_width"].to_float32() * m_font_size;
    float32 height    = glyph["bbox_height"].to_float32() * m_font_size;
    float32 bearing_x = glyph["bearing_x"].to_float32() * m_font_size;
    float32 bearing_y = glyph["bearing_y"].to_float32() * m_font_size;
    float32 advance_x = glyph["advance_x"].to_float32() * m_font_size;
    float32 x = pen[0] + bearing_x;
    float32 y = pen[1] + bearing_y;
    float32 w = width;
    float32 h = height;
    pen[0] += advance_x;

    AABB<2> texture_box;
    texture_box.m_ranges[0].include(glyph["s0"].to_float32());
    texture_box.m_ranges[0].include(glyph["s1"].to_float32());
    texture_box.m_ranges[1].include(1.f - glyph["t0"].to_float32());
    texture_box.m_ranges[1].include(1.f - glyph["t1"].to_float32());
    texture_boxs.push_back(texture_box);

    AABB<2> pixel_box;
    pixel_box.m_ranges[0].include(x);
    pixel_box.m_ranges[0].include(x+w);
    pixel_box.m_ranges[1].include(y-h);
    pixel_box.m_ranges[1].include(y);
    pixel_boxs.push_back(pixel_box);
    std::cout<<"pbox "<<pixel_box<<"\n";
    tot_aabb.include(pixel_box);
    prev_char = *it;
  }
  return tot_aabb;
}

Array<float32> Font::texture()
{
  return m_texture;
}

} // namespace dray

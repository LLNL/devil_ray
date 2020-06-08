// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/text_annotator.hpp>
#include <dray/rendering/font_factory.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>

#include<cmath>

namespace dray
{

namespace detail
{
void render_text(Array<float32> texture,
                 const int32 texture_width,
                 const int32 texture_height,
                 const int32 total_pixels,
                 Array<int32> pixel_offsets,
                 Array<AABB<2>> pboxs,
                 Array<AABB<2>> tboxs,
                 Framebuffer &fb)
{
  const int32 num_boxs = pboxs.size();
  const AABB<2> *pbox_ptr = pboxs.get_device_ptr_const();
  const AABB<2> *tbox_ptr = tboxs.get_device_ptr_const();

  const int32 *poffsets_ptr = pixel_offsets.get_device_ptr_const();
  const float32 *text_ptr = texture.get_device_ptr_const();

  const int32 width = fb.width();
  const int32 height = fb.height();

  DeviceFramebuffer d_framebuffer(fb);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_pixels), [=] DRAY_LAMBDA (int32 i)
  {
    /// figure out what pixel/box we belong to
    int box_id = num_boxs-1;
    while(i < poffsets_ptr[box_id])
    {
      box_id--;
    }
    //std::cout<<"box id "<<box_id<<"\n";
    const int local_id = i - poffsets_ptr[box_id];
    const AABB<2> pbox = pbox_ptr[box_id];
    const AABB<2> tbox = tbox_ptr[box_id];

    const float32 x_start = ceil(pbox.m_ranges[0].min());
    const float32 x_end = floor(pbox.m_ranges[0].max());
    const float32 y_start = ceil(pbox.m_ranges[1].min());
    const float32 y_end = floor(pbox.m_ranges[1].max());

    const int32 local_width = int32(x_end - x_start + 1);
    const int32 x = local_id % local_width + x_start;
    const int32 y = local_id / local_width + y_start;
    const int32 pixel_id = y * width + x;
    if(x < 0 || x >= width|| y < 0 || y >= height)
    {
      // we are outside the canvas
      return;
    }

    const float32 sx = (float32(x)-pbox.m_ranges[0].min()) / pbox.m_ranges[0].length();
    const float32 tx = (float32(y)-pbox.m_ranges[1].min()) / pbox.m_ranges[1].length();

    const float s0 = tbox.m_ranges[0].min();
    const float s1 = tbox.m_ranges[0].max();
    const float t0 = tbox.m_ranges[1].min();
    const float t1 = tbox.m_ranges[1].max();

    float32 s = lerp(s0, s1, sx) * texture_width;
    // its upside down in texture coords so invert
    float32 t = lerp(t1, t0, tx) * texture_height;

    const int32 text_id = int32(t) * texture_width + int32(s);

    float32 char_color = text_ptr[text_id];

    Vec<float32,4> color;
    color[0] = char_color;
    color[1] = char_color;
    color[2] = char_color;
    color[3] = 1.f;
    d_framebuffer.m_colors[pixel_id] = color;

    //// we need to interpolate between the two sides
    //float32 s_floor = floor(s);
    //float s_dist = s - s_floor;
    //float32 t_floor = floor(t);
    //float t_dist = t - t_floor;

    //int32 s_left = clamp(int32(s_floor), 0, texture_width-1);
    //int32 s_right = clamp(int32(ceil(s)), 0, texture_height-1);

    //float32 s_val = lerp(text_ptr[s_left], text_ptr[s_right], s_dist);

  });
  DRAY_ERROR_CHECK();
}

}// namespace detail

TextAnnotator::TextAnnotator()
  : m_font_name("MonospaceTypewriter")
{

}

void TextAnnotator::clear()
{
  m_pixel_boxs.clear();
  m_texture_boxs.clear();
}

void TextAnnotator::add_text(const std::string text,
                             const Vec<float32,2> &pos,
                             const float32 size)
{
  Font *font = FontFactory::font(m_font_name);

  font->font_size(size);

  std::vector<AABB<2>> pixel_boxs;
  std::vector<AABB<2>> texture_boxs;
  AABB<2> tot = font->font_boxs(text, pos, pixel_boxs, texture_boxs);
  m_pixel_boxs.push_back(std::move(pixel_boxs));
  m_texture_boxs.push_back(std::move(texture_boxs));
}

void TextAnnotator::render(Framebuffer &fb)
{
  // total number of characters
  int32 total_size = 0;
  for(auto box : m_pixel_boxs)
  {
    total_size += box.size();
  }

  Array<AABB<2>> pixel_boxs;
  Array<AABB<2>> texture_boxs;

  pixel_boxs.resize(total_size);
  texture_boxs.resize(total_size);

  AABB<2> *pbox_ptr = pixel_boxs.get_host_ptr();
  AABB<2> *tbox_ptr = texture_boxs.get_host_ptr();

  Array<int32> pixel_offsets;
  pixel_offsets.resize(total_size);
  int32 *pixel_offsets_ptr = pixel_offsets.get_host_ptr();

  int32 pcount = 0;
  int32 tcount = 0;
  int32 tot_pixels = 0;
  for(int32 i = 0; i < m_pixel_boxs.size(); ++i)
  {
    for(auto box : m_pixel_boxs[i])
    {
      pbox_ptr[pcount] = box;
      pixel_offsets_ptr[pcount] = tot_pixels;
      // calculate the total number of pixels
      int32 x_start = std::ceil(box.m_ranges[0].min());
      int32 x_end = std::floor(box.m_ranges[0].max());
      int32 y_start = std::ceil(box.m_ranges[1].min());
      int32 y_end = std::floor(box.m_ranges[1].max());
      int32 box_size = (x_end - x_start + 1) * (y_end - y_start + 1);
      tot_pixels += box_size;
      pcount++;
    }
    for(auto box : m_texture_boxs[i])
    {
      tbox_ptr[tcount] = box;
      tcount++;
    }
  }
  std::cout<<"Total boxs : "<<total_size<<"\n";
  std::cout<<"Total pixels : "<<tot_pixels<<"\n";

  Font *font = FontFactory::font(m_font_name);

  detail::render_text(font->texture(),
                      font->texture_width(),
                      font->texture_height(),
                      tot_pixels,
                      pixel_offsets,
                      pixel_boxs,
                      texture_boxs,
                      fb);
}

} // namespace dray

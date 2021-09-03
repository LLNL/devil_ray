// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/screen_text_annotator.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/rendering/font_factory.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/rendering/rasterbuffer.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>

#include<cmath>

namespace dray
{

namespace detail
{

DRAY_EXEC
float32 blerp(const float32 s,
              const float32 t,
              const float32 *texture,
              const int32 texture_width,
              const int32 texture_height)
{
  // we now need to blerp
  Vec<int32,2> st_min, st_max;
  st_min[0] = clamp(int32(s), 0, texture_width - 1);
  st_min[1] = clamp(int32(t), 0, texture_height - 1);
  st_max[0] = clamp(st_min[0]+1, 0, texture_width - 1);
  st_max[1] = clamp(st_min[1]+1, 0, texture_height - 1);

  Vec<float32,4> vals;
  vals[0] = texture[st_min[1] * texture_width + st_min[0]];
  vals[1] = texture[st_min[1] * texture_width + st_max[0]];
  vals[2] = texture[st_max[1] * texture_width + st_min[0]];
  vals[3] = texture[st_max[1] * texture_width + st_max[0]];

  float32 dx = s - float32(st_min[0]);
  float32 dy = t - float32(st_min[1]);

  float32 x0 = lerp(vals[0], vals[1], dx);
  float32 x1 = lerp(vals[2], vals[3], dx);
  // this the signed distance to the glyph
  return lerp(x0, x1, dy);
}

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
  Vec<float32,4> text_color = fb.foreground_color();

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

    const float32 inv_x_length = 1.f / pbox.m_ranges[0].length();
    const float32 inv_y_length = 1.f / pbox.m_ranges[1].length();
    const float32 sx = (float32(x)-pbox.m_ranges[0].min()) * inv_x_length;
    const float32 tx = (float32(y)-pbox.m_ranges[1].min()) * inv_y_length;
    const float32 sx_1 = (float32(x+1)-pbox.m_ranges[0].min()) * inv_x_length;
    const float32 tx_1 = (float32(y+1)-pbox.m_ranges[1].min()) * inv_y_length;

    const float s0 = tbox.m_ranges[0].min();
    const float s1 = tbox.m_ranges[0].max();
    const float t0 = tbox.m_ranges[1].min();
    const float t1 = tbox.m_ranges[1].max();

    float32 s = lerp(s0, s1, sx) * texture_width;
    // its upside down in texture coords so invert
    float32 t = lerp(t1, t0, tx) * texture_height;

    float32 s_1 = lerp(s0, s1, sx_1) * texture_width;
    // its upside down in texture coords so invert
    float32 t_1 = lerp(t1, t0, tx_1) * texture_height;

    float32 d_t = t_1 - t;
    float32 d_s = s_1 - s;

    // this the signed distance to the glyph
    float32 dist = detail::blerp(s,t,text_ptr,texture_width, texture_height);
    float32 d_y1 = detail::blerp(s,t_1,text_ptr,texture_width, texture_height);
    float32 d_x1 = detail::blerp(s_1,t,text_ptr,texture_width, texture_height);

    // forward difference of the distance value
    float32 dfx = d_x1 - dist;
    float32 dfy = d_x1 - dist;
    float32 width = 0.7f * sqrt(dfx*dfx + dfy*dfy);
    //float32 width = 0.1f;

    float32 alpha = smoothstep(0.5f-width,0.5f+width,dist);

    // super sample
    constexpr float32 dscale = 0.354f;
    float32 ss_dx = dscale * d_t;
    float32 ss_dy = dscale * d_s;

    float32 ss_0 = detail::blerp(s-ss_dx,t-ss_dy,text_ptr,texture_width, texture_height);
    float32 ss_1 = detail::blerp(s+ss_dx,t-ss_dy,text_ptr,texture_width, texture_height);
    float32 ss_2 = detail::blerp(s-ss_dx,t+ss_dy,text_ptr,texture_width, texture_height);
    float32 ss_3 = detail::blerp(s+ss_dx,t+ss_dy,text_ptr,texture_width, texture_height);
    ss_0 = smoothstep(0.5f-width,0.5f+width,ss_0);
    ss_1 = smoothstep(0.5f-width,0.5f+width,ss_1);
    ss_2 = smoothstep(0.5f-width,0.5f+width,ss_2);
    ss_3 = smoothstep(0.5f-width,0.5f+width,ss_3);

    alpha = (alpha + 0.5f * (ss_0 + ss_1 + ss_2 + ss_3)) / 3.f;

    Vec<float32,4> color = text_color;
    color[3] = alpha;

    Vec<float32,4> fb_color = d_framebuffer.m_colors[pixel_id];
    blend_pre_alpha(color, fb_color);
    d_framebuffer.m_colors[pixel_id] = color;

  });
  DRAY_ERROR_CHECK();
}

}// namespace detail

ScreenTextAnnotator::ScreenTextAnnotator()
  : m_font_name("OpenSans-Regular")
{

}

void ScreenTextAnnotator::clear()
{
  m_pixel_boxs.clear();
  m_texture_boxs.clear();
}

void ScreenTextAnnotator::add_text(const std::string text,
                             const Vec<float32,2> &screen_space_pos,
                             const float32 size)
{
  Font *font = FontFactory::font(m_font_name);

  font->font_size(size);

  std::vector<AABB<2>> pixel_boxs;
  std::vector<AABB<2>> texture_boxs;
  AABB<2> tot = font->font_boxs(text, screen_space_pos, pixel_boxs, texture_boxs);
  m_pixel_boxs.push_back(std::move(pixel_boxs));
  m_texture_boxs.push_back(std::move(texture_boxs));
}

void ScreenTextAnnotator::render(Framebuffer &fb)
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

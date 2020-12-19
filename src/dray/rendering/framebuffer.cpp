// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <dray/utils/png_encoder.hpp>

#include <conduit_blueprint.hpp>

namespace dray
{

Framebuffer::Framebuffer ()
: m_width (1024), m_height (1024), m_bg_color ({ 1.f, 1.f, 1.f, 1.f }),
  m_fg_color ({ 0.f, 0.f, 0.f, 1.f })
{
  m_colors.resize (m_width * m_height);
  m_depths.resize (m_width * m_height);
  clear ();
}

Framebuffer::Framebuffer (const int32 width, const int32 height)
: m_width (width), m_height (height), m_bg_color ({ 1.f, 1.f, 1.f, 1.f }),
  m_fg_color ({ 0.f, 0.f, 0.f, 1.f })
{
  assert (m_width > 0);
  assert (m_height > 0);
  m_colors.resize (m_width * m_height);
  m_depths.resize (m_width * m_height);
  clear ();
}

int32 Framebuffer::width () const
{
  return m_width;
}

int32 Framebuffer::height () const
{
  return m_height;
}

void Framebuffer::save (const std::string name)
{
  PNGEncoder png_encoder;

  png_encoder.encode ((float *)m_colors.get_host_ptr (), m_width, m_height);

  png_encoder.save (name + ".png");
}

void Framebuffer::save_depth (const std::string name)
{

  int32 image_size = m_width * m_height;

  const float32 *depth_ptr = m_depths.get_device_ptr_const ();

  RAJA::ReduceMin<reduce_policy, float32> min_val (infinity32 ());
  RAJA::ReduceMax<reduce_policy, float32> max_val (neg_infinity32 ());

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, image_size), [=] DRAY_LAMBDA (int32 i) {
    const float32 depth = depth_ptr[i];
    if (depth != infinity32 ())
    {
      min_val.min (depth);
      max_val.max (depth);
    }
  });
  DRAY_ERROR_CHECK();

  float32 minv = min_val.get ();
  float32 maxv = max_val.get ();
  const float32 len = maxv - minv;

  Array<float32> dbuffer;
  dbuffer.resize (image_size * 4);

  float32 *d_ptr = dbuffer.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, image_size), [=] DRAY_LAMBDA (int32 i) {
    const float32 depth = depth_ptr[i];
    float32 value = 0.f;

    if (depth != infinity32 ())
    {
      value = (depth - minv) / len;
    }
    const int32 offset = i * 4;
    d_ptr[offset + 0] = value;
    d_ptr[offset + 1] = value;
    d_ptr[offset + 2] = value;
    d_ptr[offset + 3] = 1.f;
  });
  DRAY_ERROR_CHECK();

  PNGEncoder png_encoder;

  png_encoder.encode (dbuffer.get_host_ptr (), m_width, m_height);

  png_encoder.save (name + ".png");
}

void Framebuffer::background_color (const Vec<float32, 4> &color)
{
  m_bg_color = color;
}

void Framebuffer::foreground_color (const Vec<float32, 4> &color)
{
  m_fg_color = color;
}

Vec<float32, 4> Framebuffer::foreground_color () const
{
  return m_fg_color;
}

Vec<float32, 4> Framebuffer::background_color () const
{
  return m_bg_color;
}

void Framebuffer::clear (const Vec<float32, 4> &color)
{
  const int32 size = m_colors.size ();
  Vec<float32, 4> clear_color = color;

  Vec<float32, 4> *color_ptr = m_colors.get_device_ptr ();
  float32 *depth_ptr = m_depths.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii) {
    depth_ptr[ii] = infinity<float32> ();
    color_ptr[ii] = clear_color;
  });
  DRAY_ERROR_CHECK();
}

void Framebuffer::clear ()
{
  clear ({ 0.f, 0.f, 0.f, 0.f });
}

void Framebuffer::composite_background ()
{
  // avoid lambda capture issues
  Vec4f background = m_bg_color;
  Vec4f *img_ptr = m_colors.get_device_ptr ();
  const int32 size = m_colors.size ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    Vec4f color = img_ptr[i];
    if (color[3] < 1.f)
    {
      blend_pre_alpha(color, background);
      img_ptr[i] = color;
    }
  });
  DRAY_ERROR_CHECK();
}

void Framebuffer::tone_map()
{
  // avoid lambda capture issues
  Vec4f *img_ptr = m_colors.get_device_ptr ();
  const int32 size = m_colors.size ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    Vec4f color = img_ptr[i];

    // ACESFilm
    constexpr float a = 2.51f;
    constexpr float b = 0.03f;
    constexpr float c = 2.43f;
    constexpr float d = 0.59f;
    constexpr float e = 0.14f;
    for(int comp = 0; comp < 3; ++comp)
    {
      float32 x = color[comp];
      x = ( ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e ) );
      x = clamp(x, 0.f,1.f);
      color[comp] = x;
    }

    img_ptr[i] = color;
  });
  DRAY_ERROR_CHECK();
}

Array<Vec<float32,4>>& Framebuffer::colors()
{
  return m_colors;
}

Array<float32>& Framebuffer::depths()
{
  return m_depths;
}


void Framebuffer::to_node(conduit::Node &mesh)
{
  mesh.reset();
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = m_width + 1;
  mesh["coordsets/coords/dims/j"] = m_height + 1;

  mesh["topologies/topo/coordset"] = "coords";
  mesh["topologies/topo/type"] = "uniform";

  const int32 size = m_colors.size();
  std::vector<float32> red, green, blue;
  red.resize(size);
  green.resize(size);
  blue.resize(size);

  for(int32 i = 0; i < size; ++i)
  {
    Vec<float32,4> color = m_colors.get_value(i);
    red[i] = color[0];
    green[i] = color[1];
    blue[i] = color[2];
  }

  mesh["fields/red/association"] = "element";
  mesh["fields/red/topology"] = "topo";
  mesh["fields/red/values"].set(red);

  mesh["fields/green/association"] = "element";
  mesh["fields/green/topology"] = "topo";
  mesh["fields/green/values"].set(green);

  mesh["fields/blue/association"] = "element";
  mesh["fields/blue/topology"] = "topo";
  mesh["fields/blue/values"].set(blue);

  mesh["fields/depth/association"] = "element";
  mesh["fields/depth/topology"] = "topo";
  const float32 *depths = m_depths.get_host_ptr_const();
  mesh["fields/depth/values"].set(depths, size);

  conduit::Node verify_info;
  bool ok = conduit::blueprint::mesh::verify(mesh,verify_info);
  if(!ok)
  {
    verify_info.print();
  }
}

} // namespace dray

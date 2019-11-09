#include <dray/framebuffer.hpp>
#include <dray/policies.hpp>

namespace dray
{

Framebuffer::Framebuffer()
  : m_width(1024),
    m_height(1024),
    m_bg_color({0.f, 0.f, 0.f, 0.f}),
    m_fg_color({0.f, 0.f, 0.f, 0.f})
{
  m_colors.resize(m_width * m_height);
  m_depths.resize(m_width * m_height);
}

Framebuffer::Framebuffer(const int32 width, const int32 height)
  : m_width(width),
    m_height(height),
    m_bg_color({0.f, 0.f, 0.f, 0.f}),
    m_fg_color({0.f, 0.f, 0.f, 0.f})
{
  assert(m_width > 0);
  assert(m_height > 0);
  m_colors.resize(m_width * m_height);
  m_depths.resize(m_width * m_height);
}

int32
Framebuffer::width() const
{
  return m_width;
}

int32
Framebuffer::height() const
{
  return m_height;
}

void
Framebuffer::background_color(const Vec<float32,4> &color)
{
  m_bg_color = color;
}

void
Framebuffer::foreground_color(const Vec<float32,4> &color)
{
  m_fg_color = color;
}

Vec<float32,4>
Framebuffer::foreground_color() const
{
  return m_fg_color;
}

Vec<float32,4>
Framebuffer::background_color() const
{
  return m_bg_color;
}

void
Framebuffer::clear(const Vec<float32,4> &color)
{
  const int32 size = m_colors.size();
  Vec<float32,4> clear_color = color;

  Vec<float32,4> *color_ptr = m_colors.get_device_ptr();
  float32 *depth_ptr = m_depths.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    depth_ptr[ii] = infinity<float32>();
    color_ptr[ii] = clear_color;
  });
}

void
Framebuffer::composite_background()
{
  // avoid lambda capture issues
  Vec4f background = m_bg_color;
  Vec4f *img_ptr = m_colors.get_device_ptr();
  const int32 size = m_colors.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Vec4f color = img_ptr[i];
    if(color[3] < 1.f)
    {
      //composite
      float32 alpha = background[3] * (1.f - color[3]);
      color[0] = color[0] + background[0] * alpha;
      color[1] = color[1] + background[1] * alpha;
      color[2] = color[2] + background[2] * alpha;
      color[3] = alpha + color[3];
      img_ptr[i] = color;
    }
  });
}


} // namespace dray

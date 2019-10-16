#include <dray/framebuffer.hpp>
#include <dray/policies.hpp>

namespace dray
{

Framebuffer::Framebuffer()
  : m_width(1024),
    m_height(1024)
{
  m_colors.resize(m_width * m_height);
  m_depths.resize(m_width * m_height);
}

Framebuffer::Framebuffer(const int32 width, const int32 height)
  : m_width(width),
    m_height(height)
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


} // namespace dray

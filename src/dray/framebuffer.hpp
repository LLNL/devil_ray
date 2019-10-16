#ifndef DRAY_FRAMEBUFFER_HPP
#define DRAY_FRAMEBUFFER_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/aabb.hpp>

namespace dray
{


class Framebuffer
{
protected:
  Array<Vec<float32,4>> m_colors;
  Array<float32>        m_depths;
  int32                 m_width;
  int32                 m_height;
public:
  Framebuffer();
  Framebuffer(const int32 width, const int32 height);

  int32 width() const;
  int32 height() const;

  void clear(const Vec<float32,4> &color);

  friend class DeviceFramebuffer;

};

} // namespace dray
#endif

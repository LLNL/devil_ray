// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/line_renderer.hpp>
#include <dray/math.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>
#include <dray/utils/timer.hpp>


namespace dray
{

namespace detail
{

// Depth value of 1.0f
const int64 clear_depth = 0x3F800000;
// Packed frame buffer value with color set as black and depth as 1.0f
const int64 clear_value = 0x3F800000000000FF;

DRAY_EXEC
float32 integer_part(float32 x)
{
  return floor(x);
}

DRAY_EXEC
float32 fractional_part(float32 x)
{
  return x - floor(x);
}

DRAY_EXEC
float32 reverse_fractional_part(float32 x)
{
  return 1.0f - fractional_part(x);
}

DRAY_EXEC
uint32 scale_color_component(float32 c)
{
  int32 t = int32(c * 256.0f);
  return uint32(t < 0 ? 0 : (t > 255 ? 255 : t));
}

DRAY_EXEC
uint32 pack_color(float32 r, float32 g, float32 b, float32 a)
{
  uint32 packed = (scale_color_component(r) << 24);
  packed |= (scale_color_component(g) << 16);
  packed |= (scale_color_component(b) << 8);
  packed |= scale_color_component(a);
  return packed;
}

DRAY_EXEC
uint32 pack_color(const Vec<float32,4> &color)
{
  return pack_color(color[0], color[1], color[2], color[3]);
}

DRAY_EXEC
void unpack_color(uint32 color,
                  float32& r,
                  float32& g,
                  float32& b,
                  float32& a)
{
  r = float32((color & 0xFF000000) >> 24) / 255.0f;
  g = float32((color & 0x00FF0000) >> 16) / 255.0f;
  b = float32((color & 0x0000FF00) >> 8) / 255.0f;
  a = float32((color & 0x000000FF)) / 255.0f;
}

DRAY_EXEC
void unpack_color(uint32 packedColor, Vec<float32,4> &color)
{
  unpack_color(packedColor, color[0], color[1], color[2], color[3]);
}


union PackedValue {

  struct PackedFloats
  {
    float32 color;
    float32 depth;
  } floats;

  struct PackedInts
  {
    uint32 color;
    uint32 depth;
  } ints;

  int64 raw;
}; // union PackedValue

struct DeviceRasterBuffer
{
  int32 m_width;
  int32 m_height;
  int64 *m_buffer;

  DeviceRasterBuffer() = delete;

  DeviceRasterBuffer(const int width, const int height, Array<int64> &buffer)
    : m_width(width),
      m_height(height),
      m_buffer(buffer.get_device_ptr())
  {
  }

  void write_pixel(const int x,
                   const int y,
                   const Vec<float32,4> &color,
                   const float32 &depth) const
  {
    const int32 index = y * m_width + x;
    PackedValue curr, next;
    next.ints.color = pack_color(color);
    next.floats.depth = depth;
    curr.floats.depth = infinity32();
    do
    {
      // only need this if we want to do alpha blending
      //unpack_color(current.ints.color, src_color);
      //float32 inverse_intensity = (1.0f - intensity);
      //float32 alpha = src_color[3] * inverse_intensity;
      //blended[0] = color[0] * intensity + src_color[0] * alpha;
      //blended[1] = color[1] * intensity + src_color[1] * alpha;
      //blended[2] = color[2] * intensity + src_color[2] * alpha;
      //blended[3] = alpha + intensity;
      //next.Ints.Color = pack_color(blended);
      //FrameBuffer.CompareExchange(index, &current.Raw, next.Raw);
      //RAJA::atomicCAS< atomic_policy >(T* acc, T compare, T value);

      // Replace the index with next if and only if the index is equal to curr.

      // saying that another way: we only want to want to replace the update the
      // buffer if our current view of the value is accurate. If it is, then we
      // know that our pixel is in front of the current value in the buffer. CAS
      // will check to see if the expected value (curr) is equal to what is in the buffer,
      // if its not, then we get an updated view of what has been written to the buffer
      curr.raw = RAJA::atomicCAS< atomic_policy >(&m_buffer[index], curr.raw, next.raw);
    } while (curr.floats.depth > next.floats.depth);

  }
};

class RasterBuffer
{
protected:
  Framebuffer &m_fb;
  Array<int64> m_int_buffer;
public:
  RasterBuffer() = delete;
  RasterBuffer(Framebuffer &fb)
    : m_fb(fb)
  {
    // copy the current state of the framebuffer into the
    // packed int64 buffer
    const int size = m_fb.depths().size();
    m_int_buffer.resize(size);
    int64 *int_ptr = m_int_buffer.get_device_ptr();
    const float *depth_ptr = m_fb.depths().get_device_ptr_const();
    const Vec<float32,4> *color_ptr = m_fb.colors().get_device_ptr_const();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
    {
      PackedValue packed;
      packed.ints.color = pack_color(color_ptr[i]);
      packed.floats.depth = depth_ptr[i];
      int_ptr[i] = packed.raw;
    });
  }

  // write the altered contents back
  // into the original framebuffer
  void finalize()
  {
    // shove the packed buffer back into the framebuffer they
    // gave us
    const int size = m_fb.depths().size();
    const int64 *int_ptr = m_int_buffer.get_device_ptr_const();
    float *depth_ptr = m_fb.depths().get_device_ptr();
    Vec<float32,4> *color_ptr = m_fb.colors().get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
    {
      PackedValue packed;
      packed.raw = int_ptr[i];

      Vec<float32,4> color;
      unpack_color(packed.ints.color, color);

      depth_ptr[i] = packed.floats.depth;
      color_ptr[i] = color;
    });
  }

  DeviceRasterBuffer device_buffer()
  {
    return DeviceRasterBuffer(m_fb.width(), m_fb.height(), m_int_buffer);
  }

};


} // namespace detail

void LineRenderer::render(Framebuffer &fb, Array<Vec<float32,3>> starts, Array<Vec<float32,3>> ends)
{
  detail::RasterBuffer raster(fb);
  detail::DeviceRasterBuffer d_raster = raster.device_buffer();

  const int num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  float elapsed_time;
  Timer mytimer = Timer();
  mytimer.reset();

  // Array<int> pixels_per_line;
  // pixels_per_line.resize(num_lines);

  // int *pixels_per_line_ptr = pixels_per_line.get_device_ptr();

  // // count the number of pixels in each line
  // RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  // {
  //   int x1,x2,y1,y2;
  //   x1 = start_ptr[i][0];
  //   y1 = start_ptr[i][1];
  //   x2 = end_ptr[i][0];
  //   y2 = end_ptr[i][1];

  //   int dx = abs(x2 - x1);
  //   int dy = abs(y2 - y1);

  //   if (dy > dx)
  //   {
  //     // then slope is greater than 1
  //     // so we need one pixel for every y value
  //     pixels_per_line_ptr[i] = dy + 1;
  //   }
  //   else
  //   {
  //     // then slope is less than 1
  //     // then we need one pixel for every x value
  //     pixels_per_line_ptr[i] = dx + 1;
  //   }
  // });

  // // should this prefix sum be parallelized???
  // // calculate offsets
  // Array<int> offsets;
  // offsets.resize(num_lines);
  // int *offsets_ptr = offsets.get_device_ptr();
  // offsets_ptr[0] = 0;
  // for (int i = 1; i < num_lines; i ++)
  // {
  //   offsets_ptr[i] = offsets_ptr[i - 1] + pixels_per_line_ptr[i - 1];
  // }

  // int num_pixels = offsets_ptr[num_lines - 1] + pixels_per_line_ptr[num_lines - 1];

  // // new containers for the next step's data
  // Array<int> x_values;
  // Array<int> y_values;
  // Array<Vec<float32, 4>> colors;
  // Array<float32> depths;
  // x_values.resize(num_pixels);
  // y_values.resize(num_pixels);
  // colors.resize(num_pixels);
  // depths.resize(num_pixels);
  // int *x_values_ptr = x_values.get_device_ptr();
  // int *y_values_ptr = y_values.get_device_ptr();
  // Vec<float32, 4> *colors_ptr = colors.get_device_ptr();
  // float32 *depths_ptr = depths.get_device_ptr();

  // save the colors and coordinates of the pixels to draw
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> color = {{1.f, 0.f, 0.f, 1.f}};
    float world_depth = 4.f;
    int x1,x2,y1,y2;
    x1 = start_ptr[i][0];
    y1 = start_ptr[i][1];
    x2 = end_ptr[i][0];
    y2 = end_ptr[i][1];

    int myindex = 0;

    int dx = abs(x2 - x1);
    int sx = x1 < x2 ? 1 : -1;
    int dy = -1 * abs(y2 - y1);
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy;
    while (true)
    {
      // x_values_ptr[myindex + offsets_ptr[i]] = x1;
      // y_values_ptr[myindex + offsets_ptr[i]] = y1;
      // colors_ptr[myindex + offsets_ptr[i]] = color;
      // depths_ptr[myindex + offsets_ptr[i]] = world_depth;

      d_raster.write_pixel(x1, y1, color, world_depth);

      myindex += 1;
      if (x1 == x2 && y1 == y2)
      {
        break;
      }
      int e2 = 2 * err;
      if (e2 >= dy)
      {
        err += dy;
        x1 += sx;
      }
      if (e2 <= dx)
      {
        err += dx;
        y1 += sy;
      }
    }
  });

  // // finally, render pixels
  // RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pixels), [=] DRAY_LAMBDA (int32 i)
  // {
  //   d_raster.write_pixel(x_values_ptr[i], y_values_ptr[i], colors_ptr[i], depths_ptr[i]);
  // });

  elapsed_time = mytimer.elapsed();
  std::cout << "elapsed time: " << elapsed_time << std::endl;

  // write this back to the original framebuffer
  raster.finalize();
}

} // namespace dray

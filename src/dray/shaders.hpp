#ifndef DRAY_SHADERS_HPP
#define DRAY_SHADERS_HPP

#include <dray/array.hpp>
#include <dray/color_table.hpp>
#include <dray/shading_context.hpp>
#include <dray/vec.hpp>

namespace dray
{

struct PointLightSource
{
  Vec<float32,3> m_pos;
  Vec<float32,3> m_amb;
  Vec<float32,3> m_diff;
  Vec<float32,3> m_spec;
  float32 m_spec_pow;
};

class Shader
{
public:
  static void composite_bg(dray::Array<dray::Vec<float, 4> > &color_buffer, 
                           dray::Vec<float, 4> &bg_color);
template<typename T>
static void blend(Array<Vec4f> &color_buffer,
                  ShadingContext<T> &shading_ctx);
template<typename T>
static void blend_surf(Array<Vec4f> &color_buffer,
                  ShadingContext<T> &shading_ctx);
static void set_color_table(ColorTable &color_table);
static int32 m_color_samples;

static void set_light_properties(const PointLightSource &light) { m_light = light; }
static void set_light_position(const Vec<float32,3> &pos) { m_light.m_pos = pos; }

private:
  //static Array<Vec4f> m_color_map;  // As a static member, was causing problems upon destruction.
  static ColorTable m_color_table;    // This is not an array, so should be fine.

  // Light properties.
  static PointLightSource m_light;
};

} // namespace dray

#endif

#ifndef DRAY_SHADERS_HPP
#define DRAY_SHADERS_HPP

#include <dray/array.hpp>
#include <dray/color_table.hpp>
#include <dray/shading_context.hpp>
#include <dray/vec.hpp>

namespace dray
{

class Shader
{
public:
  static void composite_bg(dray::Array<dray::Vec<float, 4> > &color_buffer, 
                           dray::Vec<float, 4> &bg_color);
template<typename T>
static void blend(Array<Vec4f> &color_buffer,
                  ShadingContext<T> &shading_ctx);
static void set_color_table(ColorTable &color_table);
static int32 m_color_samples;
private:
  static Array<Vec4f> m_color_map;
};

} // namespace dray

#endif

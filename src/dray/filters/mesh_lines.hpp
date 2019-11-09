#ifndef DRAY_MESH_LINES_HPP
#define DRAY_MESH_LINES_HPP


#include <dray/array.hpp>
#include <dray/color_table.hpp>
#include <dray/data_set.hpp>
#include <dray/framebuffer.hpp>
#include <dray/ray.hpp>

namespace dray
{

class MeshLines
{
protected:
  std::string m_field_name;
  ColorTable m_color_table;
  bool m_draw_mesh;
  bool m_draw_scalars;
  Range<float32> m_scalar_range;
  float32 m_line_thickness;
  int32 m_sub_element_grid_res;
public:
  MeshLines();

  template<typename ElemT>  // ElemT had better be a 2D element type.
  void execute(Array<Ray> &rays,
               DataSet<ElemT> &data_set,
               Framebuffer &fb);

  void set_field(const std::string field_name);

  void set_color_table(const ColorTable &color_table);

  void set_scalar_range(Range<float32> range);

  void draw_mesh(bool on);

  void set_line_thickness(float32 thickness);

  void set_sub_element_grid_res(int32 sub_element_grid_res);
};

};//namespace dray

#endif//DRAY_MESH_LINES_HPP

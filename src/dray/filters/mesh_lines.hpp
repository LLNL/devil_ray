#ifndef DRAY_MESH_LINES_HPP
#define DRAY_MESH_LINES_HPP

#include <dray/color_table.hpp>
#include <dray/data_set.hpp>

#include <dray/array.hpp>
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
public:
  MeshLines();

  template<typename T, typename ElemT>  // ElemT had better be a 2D element type.
  Array<Vec<float32,4>> execute(Array<Ray<T>> &rays, DataSet<T, ElemT> &data_set);

  void set_field(const std::string field_name);

  void set_color_table(const ColorTable &color_table);

  void set_scalar_range(Range<float32> range);

  void draw_mesh(bool on);

  void set_line_thickness(float32 thickness);
};

};//namespace dray

#endif//DRAY_MESH_LINES_HPP

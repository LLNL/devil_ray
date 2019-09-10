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
public:
  MeshLines();

  template<typename T, typename ElemT>  // ElemT had better be a 2D element type.
  Array<Vec<float32,4>> execute(Array<Ray<T>> &rays, DataSet<T, ElemT> &data_set);

  void set_field(const std::string field_name);

  void set_color_table(const ColorTable &color_table);
};

};//namespace dray

#endif//DRAY_MESH_LINES_HPP

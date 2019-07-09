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

  template<typename T>
  Array<Vec<float32,4>> execute(Array<Ray<T>> &rays, DataSet<T> &data_set);
};

};//namespace dray

#endif//DRAY_MESH_LINES_HPP

#ifndef DRAY_MESH_LINES_HPP
#define DRAY_MESH_LINES_HPP

#include <dray/GridFunction/mesh.hpp>

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/ray.hpp>

namespace dray
{

  template <typename T>
  Array<Vec<float32,4>> mesh_lines(Array<Ray<T>> rays, const Mesh<T> &mesh);

};//namespace dray

#endif//DRAY_MESH_LINES_HPP

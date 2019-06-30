#ifndef DRAY_MESH_LINES_HPP
#define DRAY_MESH_LINES_HPP

#include <dray/GridFunction/mesh.hpp>

#include <dray/data_set.hpp>

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/ray.hpp>
#include <dray/linear_bvh_builder.hpp>

#include <dray/ref_point.hpp> //TODO move to mesh_intersection.hpp/cpp

namespace dray
{

class Pseudocolor
{
public:
  template<typename T>
  Array<Vec<float32,4>> execute(Array<Ray<T>> &rays, DataSet<T> &data_set);
};

//TODO move to mesh_intersection.hpp/cpp
template <typename T>
Array<RefPoint<T,3>>
intersect_mesh_faces(Array<Ray<T>> rays, const Mesh<T> &mesh);

template <typename T>
Array<Vec<float32,4>> mesh_lines(Array<Ray<T>> rays, const Mesh<T> &mesh);

};//namespace dray

#endif//DRAY_MESH_LINES_HPP

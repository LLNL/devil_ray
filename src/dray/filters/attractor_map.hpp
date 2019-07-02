#ifndef DRAY_ATTRACTOR_MAP_HPP
#define DRAY_ATTRACTOR_MAP_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/data_set.hpp>
#include <dray/types.hpp>
#include <dray/ref_point.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>


namespace dray
{

  // This filter takes a mesh, element id, and a world-space query point.
  // It produces an image where each pixel location represents an initial guess, and the
  // pixel color represents whether the solver converged to an element-interior solution or to a
  // some point outside the element.
class AttractorMap
{
public:
  template<typename T>
  Array<Vec<float32,4>> execute(
      const Vec<T,3> world_query_point,
      const Array<RefPoint<T,3>> &guesses,
      Array<Vec<T,3>> &solutions,
      Array<int32> &iterations,
      DataSet<T> &data_set);

  // grid_depth becomes the exponent leading to grid size being a power of 2.
  // Makes it easier to find 3-tuple-valued indices from a linearized index.
  template <typename T>
  static
  Array<RefPoint<T,3>> domain_grid_3d(uint32 grid_depth_x, uint32 grid_depth_y, uint32 grid_depth_z, int32 el_id = 0);

  template <typename T>
  static
  Array<RefPoint<T,3>> domain_grid_slice_xy(uint32 grid_depth_x, uint32 grid_depth_y, T ref_z_val = 0.5, int32 el_id = 0);

  // Could also do other slices, but then have to decide whether x comes before z or after.
};


}//namespace dray






































































































#endif//DRAY_ATTRACTOR_MAP_HPP

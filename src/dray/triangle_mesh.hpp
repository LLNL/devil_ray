#ifndef DRAY_TRIANGLE_MESH_HPP
#define DRAY_TRIANGLE_MESH_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/ray.hpp>
#include <dray/intersection_context.hpp>

namespace dray
{

class TriangleMesh
{
protected:
  Array<float32>  m_coords;
  Array<int32>    m_indices;
  BVH             m_bvh;

  TriangleMesh(); 
public:
  TriangleMesh(Array<float32> &coords, Array<int32> &indices); 
  ~TriangleMesh(); 
  
  template<typename T>
  void            intersect(Ray<T> &rays);

  /**
   * @param[in] rays Rays that have already undergone the intersection test.
   *
   * \pre The Ray fields of m_dir, m_orig, m_dist, m_pixel_id, and m_hit_idx must be initialized.
   *
   * \retval intersection_ctx The intersection context for each ray.
   *   For any ray that does not intersect, the corresponding entry in m_is_valid is set to 0.
   */
  template<typename T>
  IntersectionContext<T> get_intersection_context(Ray<T> &rays);

  Array<float32>& get_coords();
  Array<int32>&   get_indices();
  AABB            get_bounds();

};

} // namespace dray

#endif

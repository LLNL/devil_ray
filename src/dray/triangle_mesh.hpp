#ifndef DRAY_TRIANGLE_MESH_HPP
#define DRAY_TRIANGLE_MESH_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>

namespace dray
{

class TriangleMesh
{
protected:
  Array<float32> m_coords;
  Array<int32>   m_indices;
public:
  TriangleMesh(); 
  ~TriangleMesh(); 
  Array<float32>& get_coords();
  Array<int32>&   get_indices();
  Array<AABB>     get_aabbs();
  AABB            get_bounds();
};

} // namespace dray

#endif

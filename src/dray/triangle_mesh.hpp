#ifndef DRAY_TRIANGLE_MESH_HPP
#define DRAY_TRIANGLE_MESH_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>

namespace dray
{

class TriangleMesh
{
protected:
  Array<float32>         m_coords;
  Array<int32>           m_indices;
  AABB                   m_bounds;
  Array<Vec<float32, 4>> m_bvh;

  TriangleMesh(); 
public:
  TriangleMesh(Array<float32> &coords, Array<int32> &indices); 
  ~TriangleMesh(); 

  Array<float32>& get_coords();
  Array<int32>&   get_indices();

};

} // namespace dray

#endif

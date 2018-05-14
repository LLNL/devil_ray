#include <dray/triangle_mesh.hpp>

#include <dray/linear_bvh_builder.hpp>
#include <dray/triangle_intersection.hpp>
#include <dray/policies.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

Array<AABB> 
get_tri_aabbs(Array<float32> &coords, Array<int32> indices)
{
  Array<AABB> aabbs;

  assert(indices.size() % 3 == 0);
  const int32 num_tris = indices.size() / 3;

  aabbs.resize(num_tris);

  const int32 *indices_ptr = indices.get_device_ptr_const();
  const float32 *coords_ptr = coords.get_device_ptr_const();
  AABB *aabb_ptr = aabbs.get_device_ptr();

  std::cout<<"number of triangles "<<num_tris<<"\n";
  std::cout<<"coords "<<coords.size()<<"\n";

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_tris), [=] DRAY_LAMBDA (int32 tri)
  {
    AABB aabb; 

    const int32 i_offset = tri * 3;

    for(int32 i = 0; i < 3; ++i)
    {
      const int32 vertex_id = indices_ptr[i_offset + i];
      const int32 v_offset = vertex_id * 3;
      Vec3f vertex; 

      for(int32 v = 0; v < 3; ++v)
      {
        vertex[v] = coords_ptr[v_offset + v];
      }
      aabb.include(vertex);
    }
      
    aabb_ptr[tri] = aabb;
    
  });


  return aabbs;
}

} // namespace detail


TriangleMesh::TriangleMesh(Array<float32> &coords, Array<int32> &indices)
  : m_coords(coords),
    m_indices(indices)
{
  Array<AABB> aabbs = detail::get_tri_aabbs(m_coords, indices);

  LinearBVHBuilder builder;
  builder.construct(aabbs);
  
}

TriangleMesh::TriangleMesh()
{
}

TriangleMesh::~TriangleMesh()
{

}

Array<float32>& 
TriangleMesh::get_coords()
{
  return m_coords;
}

Array<int32>& 
TriangleMesh::get_indices()
{
  return m_indices;
}

AABB
TriangleMesh::get_bounds()
{
  return m_bounds;
}


} // namespace dray


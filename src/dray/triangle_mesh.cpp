#include <dray/triangle_mesh.hpp>
#include <dray/policies.hpp>

#include <assert.h>

namespace dray
{

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
  RAJA::ReduceMin<reduce_policy, float32> xmin(infinity32());
  RAJA::ReduceMin<reduce_policy, float32> ymin(infinity32());
  RAJA::ReduceMin<reduce_policy, float32> zmin(infinity32());

  RAJA::ReduceMax<reduce_policy, float32> xmax(neg_infinity32());
  //RAJA::ReduceMax<reduce_policy, float32> ymax(neg_infinity32());
  RAJA::ReduceMax<reduce_policy, float32> ymax(-1000);
  RAJA::ReduceMax<reduce_policy, float32> zmax(neg_infinity32());
  std::cout<<"ZMAX "<<zmax.get()<<"\n";
  std::cout<<"ZMIN "<<zmin.get()<<"\n";
  std::cout<<"neg ZMIN "<<infinity32()<<" "<<neg_infinity32()<<"\n";
  std::cout<<"neg 64 ZMIN "<<infinity64()<<" "<<neg_infinity64()<<"\n";
  assert(m_coords.size() % 3 == 0);
  int32 num_coords = m_coords.size() / 3;

  const float32 *coords = m_coords.get_device_ptr_const();
  std::cout<<"coords size "<<num_coords<<"\n";
  std::cout<<"coords size "<<m_coords.size()<<"\n";
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_coords), [=] (int32 c)
  {
    const int32 offset = c * 3;
    Vec3f vertex; 

    for(int32 v = 0; v < 3; ++v)
    {
      vertex[v] = coords[offset + v];
    }
    xmin.min(vertex[0]);
    ymin.min(vertex[1]);
    zmin.min(vertex[2]);

    xmax.max(vertex[0]);
    ymax.max(vertex[1]);
    zmax.max(vertex[2]);

    //std::cout<<vertex<<"\n";
    //std::cout<<ymax.get()<<"\n";
  });
  
  AABB ret;
  Vec3f mins = make_vec3f(xmin.get(), ymin.get(), zmin.get());
  Vec3f maxs = make_vec3f(xmax.get(), ymax.get(), zmax.get());
  std::cout<<ymax.get()<<"\n";
  ret.include(mins);
  ret.include(maxs);
  std::cout<<ret<<"\n";
  return ret;
}

Array<AABB> 
TriangleMesh::get_aabbs()
{
  Array<AABB> aabbs;

  assert(m_indices.size() % 3 == 0);
  int32 num_tris = m_indices.size() / 3;

  aabbs.resize(m_indices.size());

  const int32 *indices = m_indices.get_device_ptr_const();
  const float32 *coords = m_coords.get_device_ptr_const();
  AABB *aabb_ptr = aabbs.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_tris), [=] (int32 tri)
  {
    AABB aabb; 
    const int32 i_offset = tri * 3;
    for(int32 i = 0; i < 3; ++i)
    {
      const int32 vertex_id = indices[i_offset + i];
      const int32 v_offset = vertex_id * 3;
      Vec3f vertex; 

      for(int32 v = 0; v < 3; ++v)
      {
        vertex[v] = coords[v_offset + v];
      }
      aabb.include(vertex);
    }
      
    aabb_ptr[tri] = aabb;

  });


  return aabbs;
}

} // namespace dray


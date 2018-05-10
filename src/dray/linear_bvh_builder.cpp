#include <dray/linear_bvh_builder.hpp>
#include <dray/morton_codes.hpp>
#include <dray/policies.hpp>

namespace dray
{

AABB reduce(Array<AABB> &aabbs)
{
  RAJA::ReduceMin<reduce_policy, float32> xmin(infinity32());
  RAJA::ReduceMin<reduce_policy, float32> ymin(infinity32());
  RAJA::ReduceMin<reduce_policy, float32> zmin(infinity32());

  RAJA::ReduceMax<reduce_policy, float32> xmax(neg_infinity32());
  RAJA::ReduceMax<reduce_policy, float32> ymax(neg_infinity32());
  RAJA::ReduceMax<reduce_policy, float32> zmax(neg_infinity32());

  const AABB *aabb_ptr = aabbs.get_device_ptr_const();
  //const AABB *aabb_ptr = aabbs.get_host_ptr_const();
  const int size = aabbs.size();
  std::cout<<"aabb size "<<size<<"\n";
    
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const AABB aabb = aabb_ptr[i];
    //std::cout<<i<<" "<<aabb<<"\n";
    xmin.min(aabb.m_x.min());
    ymin.min(aabb.m_y.min());
    zmin.min(aabb.m_z.min());

    xmax.max(aabb.m_x.max());
    ymax.max(aabb.m_y.max());
    zmax.max(aabb.m_z.max());

  });
 
  AABB res;
  Vec3f mins = make_vec3f(xmin.get(), ymin.get(), zmin.get());
  Vec3f maxs = make_vec3f(xmax.get(), ymax.get(), zmax.get());

  res.include(mins);
  res.include(maxs);
  return res;
}

Array<uint32> get_mcodes(Array<AABB> &aabbs, const AABB &bounds)
{
  Vec3f extent, inv_extent, min_coord;
  extent[0] = bounds.m_x.max() - bounds.m_x.min();
  extent[1] = bounds.m_y.max() - bounds.m_y.min();
  extent[2] = bounds.m_z.max() - bounds.m_z.min();

  min_coord[0] = bounds.m_x.min();
  min_coord[1] = bounds.m_y.min();
  min_coord[2] = bounds.m_z.min();

  for(int i = 0; i < 3; ++i)
  {
    inv_extent[i] = (extent[i] == .0f) ? 0.f : 1.f / extent[i];
  }

  const int size = aabbs.size();
  Array<uint32> mcodes;
  mcodes.resize(size);
  std::cout<<"aabb size "<<size<<"\n";

  const AABB *aabb_ptr = aabbs.get_device_ptr_const();
  uint32 *mcodes_ptr = mcodes.get_device_ptr();

  //std::cout<<aabbs.get_host_ptr_const()[0]<<"\n";
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const AABB aabb = aabb_ptr[i];
    // get the center and normalize it 
    float32 centroid_x = (aabb.m_x.center() - min_coord[0]) * inv_extent[0];  
    float32 centroid_y = (aabb.m_y.center() - min_coord[1]) * inv_extent[1];  
    float32 centroid_z = (aabb.m_z.center() - min_coord[2]) * inv_extent[2];  
    mcodes_ptr[i] = morton_3d(centroid_x, centroid_y, centroid_z);
  });

  return mcodes;
}

Array<int32> counting_array(const int32 &size, 
                            const int32 &start,
                            const int32 &step)
{
  
  Array<int32> iterator;
  iterator.resize(size);
  int32 *ptr = iterator.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    ptr[i] = start + i * step;
  });

  return iterator;
}


//
// reorder and array based on a new set of indices.
// array   [a,b,c]
// indices [1,0,2]
// result  [b,a,c]
//
template<typename T>
void reorder(Array<int32> &indices, Array<T> &array)
{
  assert(indices.size() == array.size());
  const int size = array.size();

  Array<T> temp;
  temp.resize(size);

  T *temp_ptr = temp.get_device_ptr();
  const T *array_ptr = array.get_device_ptr_const();
  const int32 *indices_ptr = indices.get_device_ptr_const();
   
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    int32 in_idx = indices_ptr[i];
    temp_ptr[i] = array_ptr[in_idx];
  });

  array = temp;
}

Array<int32> sort_mcodes(Array<uint32> &mcodes)
{
  const int size = mcodes.size();
  Array<int32> iter = counting_array(size, 0, 1);
  // TODO: create custom sort for GPU / CPU
  int32  *iter_ptr = iter.get_host_ptr(); 
  uint32 *mcodes_ptr = mcodes.get_host_ptr(); 
  std::sort(iter_ptr, 
            iter_ptr + size,
            [=](int32 i1, int32 i2)
            {
              return mcodes_ptr[i1] < mcodes_ptr[i2];
            });


  iter.summary();
  reorder(iter, mcodes);

  return iter;
}

void 
LinearBVHBuilder::construct(Array<AABB> &aabbs)
{
  AABB bounds = reduce(aabbs);
  std::cout<<"AABB bounds "<<bounds<<"\n";
  Array<uint32> mcodes = get_mcodes(aabbs, bounds);
  mcodes.summary();
  Array<int32> ids = sort_mcodes(mcodes);
  mcodes.summary();

}
  
} // namespace dray

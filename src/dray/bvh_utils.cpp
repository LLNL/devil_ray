#include <dray/bvh_utils.hpp>

#include <dray/aabb.hpp>
#include <dray/array_utils.hpp>
#include <dray/morton_codes.hpp>
#include <dray/sort.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

namespace dray
{

AABB<> reduce (const Array<AABB<>> &aabbs)
{


  RAJA::ReduceMin<reduce_policy, float32> xmin (infinity32 ());
  RAJA::ReduceMin<reduce_policy, float32> ymin (infinity32 ());
  RAJA::ReduceMin<reduce_policy, float32> zmin (infinity32 ());

  RAJA::ReduceMax<reduce_policy, float32> xmax (neg_infinity32 ());
  RAJA::ReduceMax<reduce_policy, float32> ymax (neg_infinity32 ());
  RAJA::ReduceMax<reduce_policy, float32> zmax (neg_infinity32 ());

  Timer timer;
  const AABB<> *aabb_ptr = aabbs.get_device_ptr_const ();
  DRAY_LOG_ENTRY ("reduce_setup", timer.elapsed ());
  timer.reset ();
  // const AABB<> *aabb_ptr = aabbs.get_host_ptr_const();
  const int size = aabbs.size ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    const AABB<> aabb = aabb_ptr[i];
    // std::cout<<i<<" "<<aabb<<"\n";
    xmin.min (aabb.m_ranges[0].min ());
    ymin.min (aabb.m_ranges[1].min ());
    zmin.min (aabb.m_ranges[2].min ());

    xmax.max (aabb.m_ranges[0].max ());
    ymax.max (aabb.m_ranges[1].max ());
    zmax.max (aabb.m_ranges[2].max ());
  });

  AABB<> res;
  Vec3f mins = make_vec3f (xmin.get (), ymin.get (), zmin.get ());
  Vec3f maxs = make_vec3f (xmax.get (), ymax.get (), zmax.get ());

  res.include (mins);
  res.include (maxs);
  return res;
}

Array<uint32> get_mcodes (Array<AABB<>> &aabbs, const AABB<> &bounds)
{
  Vec3f min_coord (bounds.min ());
  Vec3f extent (bounds.max () - bounds.min ());
  Vec3f inv_extent;

  for (int i = 0; i < 3; ++i)
  {
    inv_extent[i] = (extent[i] == .0f) ? 0.f : 1.f / extent[i];
  }

  const int size = aabbs.size ();
  Array<uint32> mcodes;
  mcodes.resize (size);

  const AABB<> *aabb_ptr = aabbs.get_device_ptr_const ();
  uint32 *mcodes_ptr = mcodes.get_device_ptr ();

  // std::cout<<aabbs.get_host_ptr_const()[0]<<"\n";
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    const AABB<> aabb = aabb_ptr[i];
    // get the center and normalize it
    float32 centroid_x = (aabb.m_ranges[0].center () - min_coord[0]) * inv_extent[0];
    float32 centroid_y = (aabb.m_ranges[1].center () - min_coord[1]) * inv_extent[1];
    float32 centroid_z = (aabb.m_ranges[2].center () - min_coord[2]) * inv_extent[2];
    mcodes_ptr[i] = morton_3d (centroid_x, centroid_y, centroid_z);
  });

  return mcodes;
}

Array<int32> sort_mcodes(Array<uint32> &mcodes)
{
  return sort(mcodes);
  //const int size = mcodes.size ();
  //Array<int32> iter = array_counting (size, 0, 1);
  //// TODO: create custom sort for GPU / CPU
  //int32 *iter_ptr = iter.get_host_ptr ();
  //uint32 *mcodes_ptr = mcodes.get_host_ptr ();

  //std::sort (iter_ptr, iter_ptr + size,
  //           [=] (int32 i1, int32 i2) { return mcodes_ptr[i1] < mcodes_ptr[i2]; });


  //reorder (iter, mcodes);

  //return iter;
}

float32
BVHData::surface_area(AABB<> aabb)
{
  float32 sa;
  float32 x = aabb.m_ranges[0].length();
  float32 y = aabb.m_ranges[1].length();
  float32 z = aabb.m_ranges[2].length();
  sa = 2.f * (x * y + x * z + y * z);
  return sa;
}

bool
BVHData::is_leaf(int32 node_index)
{
  return node_index >= m_inner_aabbs.size();
}

float32
BVHData::sam_helper(const int32 node)
{

  // http://ompf2.com/viewtopic.php?f=3&t=206&start=10
  // HMC: t_cost and i_cost are not known
  // these can be empirically determined, but t_cost << i_cost
  constexpr float32 t_cost = 1.0f;
  constexpr float32 i_cost = 1.0f;
  constexpr float32 primitives_per_leaf = 1;

  AABB<> aabb = m_inner_aabbs.get_value(node);
  float32 sa = surface_area(aabb);

  const int32 left_child = m_left_children.get_value(node);
  const int32 right_child = m_right_children.get_value(node);
  AABB<> left_aabb, right_aabb;

  float32 left_sah, right_sah;
  if(is_leaf(left_child))
  {
    int32 leaf_index = left_child - m_inner_aabbs.size();
    left_aabb = m_leaf_aabbs.get_value(leaf_index);
    left_sah = (surface_area(left_aabb) / sa) * i_cost * primitives_per_leaf;
  }
  else
  {
    left_sah = sam_helper(left_child);
    left_aabb = m_inner_aabbs.get_value(left_child);
  }

  if(is_leaf(right_child))
  {
    int32 leaf_index = right_child - m_inner_aabbs.size();
    right_aabb = m_leaf_aabbs.get_value(leaf_index);
    right_sah = (surface_area(right_aabb) / sa) * i_cost * primitives_per_leaf;
  }
  else
  {
    right_sah = sam_helper(right_child);
    right_aabb = m_inner_aabbs.get_value(right_child);
  }

  return t_cost +
         (surface_area(left_aabb) / sa) * left_sah +
         (surface_area(right_aabb) / sa) * right_sah;

}
// surface area metric
// a measure of the total cost of the tree
float32
BVHData::sam()
{
  // HMC better make sure the tree is built before calling
  float32 res = 0.f;
  res = sam_helper(0);
  return res;
}

void propagate_aabbs(BVHData &data)
{
  // HMC: this is an example of traversing the BVH in parallel from the
  // bottom up
  const int inner_size = data.m_inner_aabbs.size();
  const int leaf_size = data.m_leafs.size();

  Array<int32> counters;
  counters.resize(inner_size);

  array_memset_zero(counters);

  const int32 *lchildren_ptr = data.m_left_children.get_device_ptr_const();
  const int32 *rchildren_ptr = data.m_right_children.get_device_ptr_const();
  const int32 *parent_ptr = data.m_parents.get_device_ptr_const();
  const AABB<>  *leaf_aabb_ptr = data.m_leaf_aabbs.get_device_ptr_const();

  AABB<>  *inner_aabb_ptr = data.m_inner_aabbs.get_device_ptr();
  int32 *counter_ptr = counters.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, leaf_size), [=] DRAY_LAMBDA (int32 i)
  {
    int32 current_node = parent_ptr[inner_size + i];
    while(current_node != -1)
    {
      int32 old = RAJA::atomicAdd<atomic_policy>(&(counter_ptr[current_node]), 1);

      if(old == 0)
      {
        // first thread to get here kills itself
        return;
      }

      int32 lchild = lchildren_ptr[current_node];
      int32 rchild = rchildren_ptr[current_node];
      // gather the aabbs
      AABB<> aabb;
      if(lchild >= inner_size)
      {
        aabb.include(leaf_aabb_ptr[lchild - inner_size]);
      }
      else
      {
        aabb.include(inner_aabb_ptr[lchild]);
      }

      if(rchild >= inner_size)
      {
        aabb.include(leaf_aabb_ptr[rchild - inner_size]);
      }
      else
      {
        aabb.include(inner_aabb_ptr[rchild]);
      }

      inner_aabb_ptr[current_node] = aabb;

      current_node = parent_ptr[current_node];
    }

    //printf("There can be only one\n");

  });

  //AABB<> *inner = data.m_inner_aabbs.get_host_ptr();
  //std::cout<<"Root bounds "<<inner[0]<<"\n";
}

Array<Vec<float32,4>> emit(BVHData &data)
{
  const int inner_size = data.m_inner_aabbs.size();

  const int32 *lchildren_ptr = data.m_left_children.get_device_ptr_const();
  const int32 *rchildren_ptr = data.m_right_children.get_device_ptr_const();
  const int32 *parent_ptr    = data.m_parents.get_device_ptr_const();

  const AABB<>  *leaf_aabb_ptr  = data.m_leaf_aabbs.get_device_ptr_const();
  const AABB<>  *inner_aabb_ptr = data.m_inner_aabbs.get_device_ptr_const();

  Array<Vec<float32,4>> flat_bvh;
  flat_bvh.resize(inner_size * 4);

  Vec<float32,4> * flat_ptr = flat_bvh.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, inner_size), [=] DRAY_LAMBDA (int32 node)
  {
    Vec<float32,4> vec1;
    Vec<float32,4> vec2;
    Vec<float32,4> vec3;
    Vec<float32,4> vec4;

    AABB<> l_aabb, r_aabb;

    int32 lchild = lchildren_ptr[node];
    if(lchild >= inner_size)
    {
      l_aabb = leaf_aabb_ptr[lchild - inner_size];
      lchild = -(lchild - inner_size + 1);
    }
    else
    {
      l_aabb = inner_aabb_ptr[lchild];
      // do the offset now
      lchild *= 4;
    }

    int32 rchild = rchildren_ptr[node];
    if(rchild >= inner_size)
    {
      r_aabb = leaf_aabb_ptr[rchild - inner_size];
      rchild = -(rchild - inner_size + 1);
    }
    else
    {
      r_aabb = inner_aabb_ptr[rchild];
      // do the offset now
      rchild *= 4;
    }
    vec1[0] = l_aabb.m_ranges[0].min();
    vec1[1] = l_aabb.m_ranges[1].min();
    vec1[2] = l_aabb.m_ranges[2].min();

    vec1[3] = l_aabb.m_ranges[0].max();
    vec2[0] = l_aabb.m_ranges[1].max();
    vec2[1] = l_aabb.m_ranges[2].max();

    vec2[2] = r_aabb.m_ranges[0].min();
    vec2[3] = r_aabb.m_ranges[1].min();
    vec3[0] = r_aabb.m_ranges[2].min();

    vec3[1] = r_aabb.m_ranges[0].max();
    vec3[2] = r_aabb.m_ranges[1].max();
    vec3[3] = r_aabb.m_ranges[2].max();

    const int32 out_offset = node * 4;
    flat_ptr[out_offset + 0] = vec1;
    flat_ptr[out_offset + 1] = vec2;
    flat_ptr[out_offset + 2] = vec3;

    constexpr int32 isize = sizeof(int32);
    // memcopy so we do not truncate the ints
    memcpy(&vec4[0], &lchild, isize);
    memcpy(&vec4[1], &rchild, isize);
    flat_ptr[out_offset + 3] = vec4;
  });

  return flat_bvh;
}

} // namespace dray

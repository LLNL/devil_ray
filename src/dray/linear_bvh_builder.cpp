#include <dray/linear_bvh_builder.hpp>

#include <dray/array_utils.hpp>
#include <dray/bvh_utils.hpp>
#include <dray/math.hpp>
#include <dray/morton_codes.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

namespace dray
{

DRAY_EXEC int32 delta(const int32 &a,
                      const int32 &b,
                      const int32 &inner_size,
                      const uint32 *mcodes)
{
  bool tie = false;
  bool out_of_range = (b < 0 || b > inner_size);
  //still make the call but with a valid adderss
  const int32 bb = (out_of_range) ? 0 : b;
  const uint32 acode = mcodes[a];
  const uint32 bcode = mcodes[bb];
  //use xor to find where they differ
  uint32 exor = acode ^ bcode;
  tie = (exor == 0);
  //break the tie, a and b must always differ
  exor = tie ? uint32(a) ^ uint32(bb) : exor;
  int32 count = clz(exor);
  if (tie)
    count += 32;
  count = (out_of_range) ? - 1 : count;
  return count;
}

// Builds LBVH tree by setting parent and child pointers, given struct with
// sorted Morton codes
void build_lbvh_tree(BVHData &data)
{
  // http://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf
  const int32 inner_size = data.m_left_children.size();
  const int32 leaf_size = data.m_mcodes.size();

  int32 *lchildren_ptr = data.m_left_children.get_device_ptr();
  int32 *rchildren_ptr = data.m_right_children.get_device_ptr();
  int32 *parent_ptr = data.m_parents.get_device_ptr();
  const uint32 *mcodes_ptr = data.m_mcodes.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, inner_size), [=] DRAY_LAMBDA (int32 i)
  {
    //determine range direction
    int32 d = 0 > (delta(i, i + 1, inner_size, mcodes_ptr) - delta(i, i - 1, inner_size, mcodes_ptr)) ? -1 : 1;

    //find upper bound for the length of the range
    int32 min_delta = delta(i, i - d, inner_size, mcodes_ptr);
    int32 lmax = 2;
    while (delta(i, i + lmax * d, inner_size, mcodes_ptr) > min_delta)
          lmax *= 2;

    //binary search to find the lower bound
    int32 l = 0;
    for (int32 t = lmax / 2; t >= 1; t /= 2)
    {
      if (delta(i, i + (l + t) * d, inner_size, mcodes_ptr) > min_delta)
      {
        l += t;
      }
    }

    int32 j = i + l * d;
    int32 delta_node = delta(i, j, inner_size, mcodes_ptr);
    int32 s = 0;
    float32 div_factor = 2.f;
    //find the split postition using a binary search
    for (int32 t = (int32)ceil(float32(l) / div_factor);;
         div_factor *= 2, t = (int32)ceil(float32(l) / div_factor))
    {
      if (delta(i, i + (s + t) * d, inner_size, mcodes_ptr) > delta_node)
      {
        s += t;
      }
      if (t == 1) break;
    }

    int32 split = i + s * d + min(d, 0);
    // assign parent/child pointers
    if (min(i, j) == split)
    {
      //leaf
      parent_ptr[split + inner_size] = i;
      lchildren_ptr[i] = split + inner_size;
    }
    else
    {
      //inner node
      parent_ptr[split] = i;
      lchildren_ptr[i] = split;
    }


    if (max(i, j) == split + 1)
    {
      //leaf
      parent_ptr[split + inner_size + 1] =  i;
      rchildren_ptr[i] =  split + inner_size + 1;
    }
    else
    {
      parent_ptr[split + 1] = i;
      rchildren_ptr[i] = split + 1;
    }

    if(i == 0)
    {
      // flag the root
      parent_ptr[0] = -1;
    }

  });

}

BVH
LinearBVHBuilder::construct(Array<AABB<>> aabbs)
{
  Array<int32> primitive_ids = array_counting(aabbs.size(), 0, 1);
  return construct(aabbs, primitive_ids);
}

BVH
LinearBVHBuilder::construct(Array<AABB<>> aabbs, Array<int32> primitive_ids)
{
  DRAY_LOG_OPEN("bvh_construct");
  DRAY_LOG_ENTRY("num_aabbs", aabbs.size());

  if(aabbs.size() == 1)
  {
    //Special case that we have to deal with due to
    //the internal bvh representation
    Array<AABB<>> new_aabbs;
    new_aabbs.resize(2);
    AABB<> *old_ptr = nullptr, *new_ptr = nullptr;
    old_ptr = aabbs.get_host_ptr();
    new_ptr = new_aabbs.get_host_ptr();
    new_ptr[0] = old_ptr[0];
    AABB<> invalid;
    new_ptr[1] = invalid;

    aabbs = new_aabbs;
  }
  Timer tot_time;
  Timer timer;

  AABB<> bounds = reduce(aabbs);
  DRAY_LOG_ENTRY("reduce", timer.elapsed());
  timer.reset();

  Array<uint32> mcodes = get_mcodes(aabbs, bounds);
  DRAY_LOG_ENTRY("morton_codes", timer.elapsed());
  timer.reset();

  // original positions of the sorted morton codes.
  // allows us to gather / sort other arrays.
  Array<int32> ids = sort_mcodes(mcodes);
  DRAY_LOG_ENTRY("sort", timer.elapsed());
  timer.reset();

  reorder(ids, aabbs);
  reorder(ids, primitive_ids);
  DRAY_LOG_ENTRY("reorder", timer.elapsed());
  timer.reset();

  const int size = aabbs.size();

  BVHData bvh_data;
  // the arrays that already exist
  bvh_data.m_leafs  = primitive_ids;
  bvh_data.m_mcodes = mcodes;
  bvh_data.m_leaf_aabbs = aabbs;
  // arrays we have to calculate
  bvh_data.m_inner_aabbs.resize(size - 1);
  bvh_data.m_left_children.resize(size - 1);
  bvh_data.m_right_children.resize(size - 1);
  bvh_data.m_parents.resize(size + size - 1);

  // assign parent and child pointers
  build_lbvh_tree(bvh_data);
  DRAY_LOG_ENTRY("build_tree", timer.elapsed());
  timer.reset();

  propagate_aabbs(bvh_data);
  DRAY_LOG_ENTRY("propagate", timer.elapsed());
  timer.reset();

  // tree is now constructed
  // HMC: if you want to optimize the topology, this is where you would do it
  DRAY_LOG_ENTRY("sam", bvh_data.sam());
  timer.reset();

  BVH bvh;
  bvh.m_inner_nodes = emit(bvh_data);
  DRAY_LOG_ENTRY("emit", timer.elapsed());
  timer.reset();

  bvh.m_leaf_nodes = bvh_data.m_leafs;
  bvh.m_bounds = bounds;
  bvh.m_aabb_ids = ids;

  DRAY_LOG_ENTRY("tot_time", tot_time.elapsed());
  DRAY_LOG_CLOSE();
  return bvh;
}

} // namespace dray

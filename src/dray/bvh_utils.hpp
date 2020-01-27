#ifndef DRAY_BVH_UTILS_HPP
#define DRAY_BVH_UTILS_HPP

#include <dray/aabb.hpp>
#include <dray/array_utils.hpp>

namespace dray
{

// takes in array of AABBs
// calculates min/max in each dimension
// returns AABB that contains all input AABBs
AABB<> reduce(const Array<AABB<>> &aabbs);

// computes Morton codes for bounding boxes given a list of them and a bounding
// box encompassing all of them
Array<uint32> get_mcodes(Array<AABB<>> &aabbs, const AABB<> &bounds);

// reorder an array based on a new set of indices.
// array   [a,b,c]
// indices [1,0,2]
// result  [b,a,c]
template <typename T>
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

// Sorts Morton codes and returns set of indices with same ordering
Array<int32> sort_mcodes(Array<uint32> &mcodes);

struct BVHData
{
  Array<int32> m_left_children; // only inner nodes
  Array<int32> m_right_children; // only inner nodes
  // children notes: if child >= m_inner.aabbs.size() then its a leaf
  // index = child - inner_size
  Array<int32> m_parents; // everyone has parents
  Array<int32> m_leafs; // size = inner + 1
  Array<uint32> m_mcodes; // size of leafs
  Array<AABB<>> m_inner_aabbs; // size of inner nodes
  Array<AABB<>> m_leaf_aabbs; // size of leafs

  // Compute surface area of given AABB
  float32 surface_area(AABB<> aabb);

  // Determines whether given index corresponds to a leaf node
  bool is_leaf(int32 node_index);

  // Computes surface area metric starting from a given node
  float32 sam_helper(const int32 node);

  // Computes surface area heuristic of whole BVH
  float32 sam();

};

// Traverses tree and for each node, sets bounding box to be the union of the
// bounding boxes of children
// (requires that all nodes have parents (and children, if applicable) set)
void propagate_aabbs(BVHData &data);

// Constructs finalized BVH array based on intermediate representation
// (requires AABBs to be propogated)
Array<Vec<float32,4>> emit(BVHData &data);

} // namespace dray

#endif // DRAY_BVH_HELPERS_HPP

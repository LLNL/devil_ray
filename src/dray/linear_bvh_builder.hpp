#ifndef DRAY_LINEAR_BVH_BUILDER_HPP
#define DRAY_LINEAR_BVH_BUILDER_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>

namespace dray
{

struct BVH
{
  Array<Vec<float32, 4>> m_inner_nodes;
  Array<int32>           m_leaf_nodes;
  AABB<>                 m_bounds;
  Array<int32>           m_aabb_ids;
  // multiple leaf nodes can point to the same
  // original primitive. m_aabb_ids point to the
  // index of the aabb given to construct
};

class LinearBVHBuilder
{

public:
  BVH construct(Array<AABB<>> aabbs);
  BVH construct(Array<AABB<>> aabbs, Array<int32> primimitive_ids);

};

AABB<> reduce(const Array<AABB<>> &aabbs);

} // namespace dray
#endif

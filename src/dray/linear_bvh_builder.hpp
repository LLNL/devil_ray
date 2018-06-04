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
  AABB                   m_bounds;
};

class LinearBVHBuilder
{

public:
  BVH construct(Array<AABB> &aabbs);
  
};

} // namespace dray
#endif

#ifndef DRAY_LINEAR_BVH_BUILDER_HPP
#define DRAY_LINEAR_BVH_BUILDER_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>

namespace dray
{

class LinearBVHBuilder
{

public:
  void construct(Array<AABB> &aabbs);
  
};

} // namespace dray
#endif

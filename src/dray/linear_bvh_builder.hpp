#ifndef DRAY_LINEAR_BVH_BUILDER_HPP
#define DRAY_LINEAR_BVH_BUILDER_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>
#include <dray/bvh.hpp>

namespace dray
{

class LinearBVHBuilder
{

public:
  BVH construct(Array<AABB<>> aabbs);
  BVH construct(Array<AABB<>> aabbs, Array<int32> primimitive_ids);
};

AABB<> reduce(const Array<AABB<>> &aabbs);

} // namespace dray
#endif

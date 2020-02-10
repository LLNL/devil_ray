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
  // Construct BVH given AABBs of primitives
  // (this is the simple case where each AABB corresponds to one primitive)
  BVH construct(Array<AABB<>> aabbs);

  // Construct BVH given AABBs of primitives and mapping of bounding boxes to
  // un-subdivided primitive IDs
  BVH construct(Array<AABB<>> aabbs, Array<int32> primimitive_ids);

};

AABB<> reduce(const Array<AABB<>> &aabbs);

} // namespace dray
#endif

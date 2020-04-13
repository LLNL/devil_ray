#ifndef DRAY_AAC_BUILDER_HPP
#define DRAY_AAC_BUILDER_HPP

#include <dray/aabb.hpp>
#include <dray/bvh.hpp>

namespace dray
{

class Cluster
{

public:

  enum ClusterType {Root, LeftChild, RightChild};

  Cluster();
  ~Cluster();

  Cluster *closest;
  Cluster *left;
  Cluster *right;

  Cluster *parent;
  ClusterType cluster_type;
  int aabb_id; // index into aabbs array from construct
  AABB<> aabb;
  int prim_id; // index into primitive_ids array from construct FIXME: i think we can delete this

  bool isLeaf();
};

class AACBuilder
{

public:
  // Construct BVH given AABBs of primitives
  // (this is the simple case where each AABB corresponds to one primitive)
  BVH construct(Array<AABB<>> aabbs);

  // Construct BVH given AABBs of primitives and mapping of bounding boxes to
  // un-subdivided primitive IDs
  BVH construct(Array<AABB<>> aabbs, Array<int32> primimitive_ids);

};

} // namespace dray

#endif // DRAY_AAC_BUILDER_HP

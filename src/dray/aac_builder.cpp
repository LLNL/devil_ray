#include <dray/aac_builder.hpp>

#include <dray/array_utils.hpp>
#include <dray/bvh_utils.hpp>
#include <dray/math.hpp>
#include <dray/morton_codes.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

// FIXME: these should not be constants
#define AAC_DELTA 20 // 20 for HQ, 4 for fast
#define AAC_EPSILON 0.1f //0.1f for HQ, 0.2 for fast
#define AAC_ALPHA (0.5f-AAC_EPSILON)

// 30 bit mcodes
#define MORTON_CODE_END 0
#define MORTON_CODE_START 29

// FIXME: get rid of centroids

namespace dray
{

static inline float AAC_C() {
    return (0.5f * powf(AAC_DELTA, 0.5f + AAC_EPSILON));
}

static inline uint32_t AAC_F(uint32_t x)
{
    return (uint32_t) (ceil(AAC_C() * powf(x, AAC_ALPHA)));
}

Array<uint32> get_mcodes (Array<AABB<>> &aabbs, const AABB<> &bounds, Array<Vec3f> &centroids)
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
  centroids.resize (size);

  const AABB<> *aabb_ptr = aabbs.get_device_ptr_const ();
  uint32 *mcodes_ptr = mcodes.get_device_ptr ();
  Vec3f *centroids_ptr = centroids.get_device_ptr ();

  // std::cout<<aabbs.get_host_ptr_const()[0]<<"\n";
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    const AABB<> aabb = aabb_ptr[i];
    // get the center and normalize it
    float32 centroid_x = (aabb.m_ranges[0].center () - min_coord[0]);
    float32 centroid_y = (aabb.m_ranges[1].center () - min_coord[1]);
    float32 centroid_z = (aabb.m_ranges[2].center () - min_coord[2]);
    centroids_ptr[i] = {centroid_x, centroid_y, centroid_z};
    centroid_x *= inv_extent[0];
    centroid_y *= inv_extent[1];
    centroid_z *= inv_extent[2];
    mcodes_ptr[i] = morton_3d (centroid_x, centroid_y, centroid_z);
  });

  return mcodes;
}

Cluster::Cluster()
  : closest{nullptr}, left{nullptr}, right{nullptr}, parent{nullptr}, cluster_type{Root}, prim_id{-1}
{
  // nothing else to do
}

Cluster::~Cluster()
{
  if (left != nullptr)
    delete left;

  if (right != nullptr)
    delete right;

}

bool Cluster::isLeaf() {
  return (this->left == nullptr && this->right == nullptr);
}


size_t
makePartition(const Array<uint32> &mcodes, size_t start, size_t end,
    size_t partitionbit)
{

  const uint32 *mcodes_ptr = mcodes.get_device_ptr_const();

  size_t curFind = (1 << partitionbit);

  if (((mcodes_ptr[start] & curFind) == (mcodes_ptr[end-1] & curFind))
      || (partitionbit < MORTON_CODE_END))
  {
      return start + (end-start)/2;
  }

  size_t lower = start;
  size_t upper = end;

  while (lower < upper)
  {
      size_t mid = lower + (upper-lower)/2;
      if ((mcodes_ptr[mid] & curFind) == 0)
      {
          lower = mid+1;
      } else
      {
          upper = mid;
      }
  }

  return lower;
}

size_t
findBestMatch(const Array<AABB<>> &aabbs, const std::vector<Cluster *> &clusters, size_t i)
{
  const AABB<> *aabbs_ptr = aabbs.get_device_ptr_const();

  float closestDist = std::numeric_limits<float>::infinity();
  size_t idx = i;
  for (size_t j = 0; j < clusters.size(); ++j) {
    if (i == j) continue;

    AABB<> combined = (clusters[i]->aabb).intersect(clusters[j]->aabb);
    float d = combined.area();
    if (d < closestDist) {
      closestDist = d;
      idx = j;
    }
  }

  return idx;
}

void
combineClusters(const Array<AABB<>> &aabbs, std::vector<Cluster *> &clusters, size_t n, size_t &total_nodes,
    size_t dim)
{

  std::vector<size_t> closest(clusters.size(), 0);

  for (size_t i = 0; i < clusters.size(); ++i)
  {
    closest[i] = findBestMatch(aabbs, clusters, i);
  }

  while (clusters.size() > n)
  {
    float best_dist = std::numeric_limits<float>::infinity();
    size_t left_idx = 0;
    size_t right_idx = 0;

    const AABB<> *aabbs_ptr = aabbs.get_device_ptr_const();

    for (size_t i = 0; i < clusters.size(); ++i)
    {
  //    BBox combined = Union(clusters[i]->bounds, clusters[closest[i]]->bounds);
      const AABB<> combined = (clusters[i] -> aabb).onion(clusters[closest[i]] -> aabb);
      float d = combined.area();
      if (d < best_dist)
      {
        best_dist = d;
        left_idx = i;
        right_idx = closest[i];
      }
    }

    ++total_nodes;
    Cluster *node = new Cluster();

  //  node->InitInterior(dim, clusters[leftIdx], clusters[rightIdx]);
    clusters[left_idx]->parent = node;
    clusters[left_idx]->cluster_type = Cluster::LeftChild;

    clusters[right_idx]->parent = node;
    clusters[right_idx]->cluster_type = Cluster::RightChild;

    node->left = clusters[left_idx];
    node->right = clusters[right_idx];
    node->aabb = (node->left->aabb).onion(node->right->aabb);

    clusters[left_idx] = node;
    assert(node != nullptr);
    clusters[right_idx] = clusters.back();
    assert(clusters[right_idx] != nullptr);
    closest[right_idx] = closest.back();
    clusters.pop_back();
    closest.pop_back();
    closest[left_idx] = findBestMatch(aabbs, clusters, left_idx);

    for (size_t i = 0; i < clusters.size(); ++i)
    {
      if (closest[i] == left_idx || closest[i] == right_idx)
      {
        closest[i] = findBestMatch(aabbs, clusters, i);
      } else if (closest[i] == closest.size())
      {
        closest[i] = right_idx;
      }
    }
  }

}

// FIXME: document
// clusters is JUST the output
void
buildTree(const Array<AABB<>> &aabbs, const Array<int32> &primitive_ids,
    const Array<uint32> &mcodes, const Array<Vec3f> &centroids, size_t start,
    size_t end, size_t &total_nodes, size_t partition_bit,
    std::vector<Cluster*> &clusters)
{

  if (end-start == 0)
  {
      return;
  }

  assert(clusters.size() == 0);

  int dim = partition_bit % 3;

  if (end-start < AAC_DELTA)
  {
    //std::cout << "base case of buildTree, |P| = " << end-start << "\n";
    //std::vector<Cluster *> prim_clusters;
    //prim_clusters.resize(end-start);
    //clusters.resize(end-start);

    total_nodes += (end-start);

    const int32 *primitive_ids_ptr = primitive_ids.get_device_ptr_const();

    const AABB<> *aabbs_ptr = aabbs.get_device_ptr_const();

    for (size_t i = start; i < end; ++i)
    {
        // Create leaf cluster
        Cluster *node = new Cluster();
        node->aabb_id = i;
        node->aabb = aabbs_ptr[i];
        node->prim_id = primitive_ids_ptr[i];
        clusters.push_back(node);
        assert(node != nullptr);
    }

    //*clusterData = combineCluster(clusters, AAC_F(AAC_DELTA), totalNodes, dim);
    combineClusters(aabbs, clusters, AAC_F(AAC_DELTA), total_nodes, dim);

    return;
  }

  //std::cout << "general case of buildTree, |P| = " << end-start << "\n";

  //std::cout << "{\n";

  size_t splitIdx = makePartition(mcodes, start, end, partition_bit);

  size_t new_partition_bit = partition_bit - 1;
  std::vector<Cluster *> left_clusters;
  std::vector<Cluster *> right_clusters;
  size_t right_total_nodes = 0;

  buildTree(aabbs, primitive_ids, mcodes, centroids, start, splitIdx, total_nodes, new_partition_bit, left_clusters);
  //std::cout << "left buildTree gave us " << left_clusters.size() << " clusters\n";
  buildTree(aabbs, primitive_ids, mcodes, centroids, splitIdx, end, right_total_nodes, new_partition_bit, right_clusters);
  //std::cout << "right buildTree gave us " << right_clusters.size() << " clusters\n";

  total_nodes += right_total_nodes;

  left_clusters.insert( left_clusters.end(), right_clusters.begin(), right_clusters.end() );
  //*clusterData = combineCluster(leftC, AAC_F(end-start), totalNodes, dim);
  combineClusters(aabbs, left_clusters, AAC_F(end-start), total_nodes, dim);

  clusters = left_clusters;
  //std::cout << "}\n";
}

//Array<Vec<float32,4>>
//emit(const Array<AABB<>> &aabbs, Cluster &root)
//{
//  const int inner_size = data.m_inner_aabbs.size();
//
//  const int32 *lchildren_ptr = data.m_left_children.get_device_ptr_const();
//  const int32 *rchildren_ptr = data.m_right_children.get_device_ptr_const();
//  const int32 *parent_ptr    = data.m_parents.get_device_ptr_const();
//
//  const AABB<>  *leaf_aabb_ptr  = data.m_leaf_aabbs.get_device_ptr_const();
//  const AABB<>  *inner_aabb_ptr = data.m_inner_aabbs.get_device_ptr_const();
//
//  Array<Vec<float32,4>> flat_bvh;
//  flat_bvh.resize(inner_size * 4);
//
//  Vec<float32,4> * flat_ptr = flat_bvh.get_device_ptr();
//
//  RAJA::forall<for_policy>(RAJA::RangeSegment(0, inner_size), [=] DRAY_LAMBDA (int32 node)
//  {
//    Vec<float32,4> vec1;
//    Vec<float32,4> vec2;
//    Vec<float32,4> vec3;
//    Vec<float32,4> vec4;
//
//    AABB<> l_aabb, r_aabb;
//
//    int32 lchild = lchildren_ptr[node];
//    if(lchild >= inner_size)
//    {
//      l_aabb = leaf_aabb_ptr[lchild - inner_size];
//      lchild = -(lchild - inner_size + 1);
//    }
//    else
//    {
//      l_aabb = inner_aabb_ptr[lchild];
//      // do the offset now
//      lchild *= 4;
//    }
//
//    int32 rchild = rchildren_ptr[node];
//    if(rchild >= inner_size)
//    {
//      r_aabb = leaf_aabb_ptr[rchild - inner_size];
//      rchild = -(rchild - inner_size + 1);
//    }
//    else
//    {
//      r_aabb = inner_aabb_ptr[rchild];
//      // do the offset now
//      rchild *= 4;
//    }
//    vec1[0] = l_aabb.m_ranges[0].min();
//    vec1[1] = l_aabb.m_ranges[1].min();
//    vec1[2] = l_aabb.m_ranges[2].min();
//
//    vec1[3] = l_aabb.m_ranges[0].max();
//    vec2[0] = l_aabb.m_ranges[1].max();
//    vec2[1] = l_aabb.m_ranges[2].max();
//
//    vec2[2] = r_aabb.m_ranges[0].min();
//    vec2[3] = r_aabb.m_ranges[1].min();
//    vec3[0] = r_aabb.m_ranges[2].min();
//
//    vec3[1] = r_aabb.m_ranges[0].max();
//    vec3[2] = r_aabb.m_ranges[1].max();
//    vec3[3] = r_aabb.m_ranges[2].max();
//
//    const int32 out_offset = node * 4;
//    flat_ptr[out_offset + 0] = vec1;
//    flat_ptr[out_offset + 1] = vec2;
//    flat_ptr[out_offset + 2] = vec3;
//
//    constexpr int32 isize = sizeof(int32);
//    // memcopy so we do not truncate the ints
//    memcpy(&vec4[0], &lchild, isize);
//    memcpy(&vec4[1], &rchild, isize);
//    flat_ptr[out_offset + 3] = vec4;
//  });
//
//  return flat_bvh;
//}

Array<Vec<float32, 4>>
emit(const Array<AABB<>> &aabbs, Cluster *root_node)
{
  const int leaf_nodes = aabbs.size();
  const int inner_size = leaf_nodes - 1;

  Array<Vec<float32,4>> flat_bvh;
  flat_bvh.resize(inner_size * 4);
  Vec<float32,4> *flat_bvh_ptr = flat_bvh.get_device_ptr();

  // populate flat_bvh data structure
  std::stack<Cluster *> todo;
  todo.push(root_node);

  std::map<Cluster *, size_t> cluster_indices;

  size_t array_index = 0;
  while (!todo.empty())
  {
    Cluster *this_node = todo.top();
    todo.pop();
    //std::cout << "visiting node..." << std::endl;

    if (this_node->cluster_type != Cluster::Root)
    {
      auto it = cluster_indices.find(this_node -> parent);

      assert(it != cluster_indices.end());
      size_t parent_idx = it->second;

      // left comes before right
      // we are definitely not a root node per the check above so we are either a left or a right
      size_t offset = (this_node->cluster_type == Cluster::LeftChild) ? 0 : 1;

      // these next elements are 6 floats wide with left holding the first 6 and right holding the second 6
      size_t aabb_offset = offset * 6;

      // add ranges to parent struct
      // for each of left and right, we have the following 6 floats:
      // [xmin, ymin, zmin, xmax, ymax, zmax]
      // so we extract the range and iterate from 0 to 3 and populate the imin, (i+3)max values
      for (size_t i = 0; i < 3; ++i)
      {
        const Range &r = this_node->aabb.m_ranges[i];

        size_t min_offset = aabb_offset + i;
        size_t max_offset = aabb_offset + 3 + i;

        flat_bvh_ptr[parent_idx * 4 + min_offset/4][min_offset % 4] = r.min();
        flat_bvh_ptr[parent_idx * 4 + max_offset/4][max_offset % 4] = r.max();
      }

      int current_index = 0;

      if (this_node->isLeaf())
      {
        // this is a leaf node
        current_index = -(this_node->aabb_id + 1);
      } else
      {
        // this is an internal node
        cluster_indices.insert({this_node, array_index});

        current_index = array_index;
      }

      flat_bvh_ptr[parent_idx * 4 + 3][offset] = reinterpret_cast<float32 &>(current_index);
    }

    // skip inserting leaf nodes, and also skip adding their children
    if (!(this_node->isLeaf()))
    {
      cluster_indices.insert({this_node, array_index});
      ++array_index;

      todo.push(this_node->left);
      todo.push(this_node->right);
    }

  }

  //std::cout << "digraph G {\n";

  //for (size_t i = 0; i < flat_bvh.size()/4; ++i)
  //{
  //  std::cout << i << " -> " << reinterpret_cast<int &>(flat_bvh_ptr[i*4 + 3][0]) << "\n";
  //  std::cout << i << " -> " << reinterpret_cast<int &>(flat_bvh_ptr[i*4 + 3][1]) << "\n";
  //}

  //std::cout << "}" << std::endl;

  return flat_bvh;
}

size_t getTreeSize(Cluster *root)
{
  size_t accumulator = 1;
  if (root -> left != nullptr)
    accumulator += getTreeSize(root->left);

  if (root -> right != nullptr)
    accumulator += getTreeSize(root->right);

  return accumulator;

}

std::string getName(Cluster *node)
{
  long long address = (long long) node;
  return std::to_string(address) + ", " + std::to_string(node->cluster_type) + ", " + std::to_string(node->isLeaf());

}

// helper function for printing tree using graphviz
void printTree(Cluster *root)
{
  if (root->cluster_type != Cluster::Root)
  {
    std::cout << "\"" << getName(root) << "\" -> \"" << getName(root->parent) << "\"\n";
  }
  if (!root->isLeaf())
  {
    std::cout << "\"" << getName(root) << "\" -> \"" << getName(root->left) << "\"\n";
    std::cout << "\"" << getName(root) << "\" -> \"" << getName(root->right) << "\"\n";

    printTree(root->left);
    printTree(root->right);
  }
}

BVH
AACBuilder::construct(Array<AABB<>> aabbs)
{
  Array<int32> primitive_ids = array_counting(aabbs.size(), 0, 1);
  return construct(aabbs, primitive_ids);
}

BVH
AACBuilder::construct(Array<AABB<>> aabbs, Array<int32> primitive_ids)
{
  DRAY_LOG_OPEN("bvh_construct");
  DRAY_LOG_ENTRY("num_aabbs", aabbs.size());
  std::cout << "aabbs:" << aabbs.size() << std::endl;

  Timer tot_time;
  Timer timer;

  AABB<> bounds = reduce(aabbs);
  DRAY_LOG_ENTRY("reduce", timer.elapsed());
  timer.reset();

  Array<Vec3f> centroids;
  Array<uint32> mcodes = get_mcodes(aabbs, bounds, centroids);
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

  // Build AAC
  std::vector<Cluster *> clusters;
  size_t total_nodes = 0;
  buildTree(aabbs, primitive_ids, mcodes, centroids, 0, aabbs.size(),
      total_nodes, 0, clusters);


  std::cout << "getTreeSize: " << getTreeSize(clusters[0]) << std::endl;
  std::cout << "total_nodes: " << total_nodes << std::endl;

  printTree(clusters[0]);

  // Emit expected structure
  BVH bvh;
  bvh.m_inner_nodes = emit(aabbs, clusters[0]);
  DRAY_LOG_ENTRY("emit", timer.elapsed());
  timer.reset();

  bvh.m_leaf_nodes = primitive_ids;
  bvh.m_bounds = bounds;
  bvh.m_aabb_ids = ids;

  DRAY_LOG_ENTRY("tot_time", tot_time.elapsed());
  DRAY_LOG_CLOSE();

  return bvh;
}

} // namespace dray

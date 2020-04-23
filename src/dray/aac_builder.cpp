#include <dray/aac_builder.hpp>

#include <dray/array_utils.hpp>
#include <dray/bvh_utils.hpp>
#include <dray/math.hpp>
#include <dray/morton_codes.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>
#include <queue>

// FIXME: these should not be constants
#define AAC_DELTA 20 // 20 for HQ, 4 for fast
#define AAC_EPSILON 0.1f //0.1f for HQ, 0.2 for fast
#define AAC_ALPHA (0.5f-AAC_EPSILON)

// 30 bit mcodes
#define MORTON_CODE_END 0
#define MORTON_CODE_START 29

namespace dray
{

static inline float AAC_C() {
    return (0.5f * powf(AAC_DELTA, 0.5f + AAC_EPSILON));
}

static inline uint32_t AAC_F(uint32_t x)
{
    return (uint32_t) (ceil(AAC_C() * powf(x, AAC_ALPHA)));
}

float32 surface_area(AABB<> aabb)
{
  float32 sa;
  float32 x = aabb.m_ranges[0].length();
  float32 y = aabb.m_ranges[1].length();
  float32 z = aabb.m_ranges[2].length();
  sa = 2.f * (x * y + x * z + y * z);
  return sa;
}

// Internal representation of tree structure for building AAC
class Cluster
{

public:

  // Used for combineClusters
  Cluster *closest;

  // Children pointers
  Cluster *left;
  Cluster *right;

  Cluster *parent;

  // If this cluster does not have a parent, it is a root
  // If it does, it is either a Left or Right child depending on which the
  // parent thinks it is
  enum ClusterType {Root, LeftChild, RightChild};
  ClusterType cluster_type;

  // index into aabbs array from construct
  int aabb_id;

  AABB<> aabb;

  Cluster() = delete;

  // Construct cluster with primitive
  Cluster(AABB<> aabb, int aabb_id);

  // Construct cluster with left and right children
  Cluster(AABB<> aabb, Cluster *left, Cluster *right);

  ~Cluster();

  bool isLeaf() const;
};



Cluster::Cluster(AABB<> aabb_, int aabb_id_)
  : closest{nullptr}, left{nullptr}, right{nullptr}, parent{nullptr},
  cluster_type{Root}, aabb_id{aabb_id_}
{
  // I'm not really sure why we can't put this in the initializer list but
  // the compiler complained so here we are
  aabb = aabb_;
}

Cluster::Cluster(AABB<> aabb_, Cluster *left_, Cluster *right_)
  : closest{nullptr}, left{left_}, right{right_}, parent{nullptr},
  cluster_type{Root}, aabb_id{-1}
{
  // I'm not really sure why we can't put this in the initializer list but
  // the compiler complained so here we are
  aabb = aabb_;
}

Cluster::~Cluster()
{
  if (left != nullptr)
    delete left;

  if (right != nullptr)
    delete right;

}

bool Cluster::isLeaf() const {
  return (this->left == nullptr && this->right == nullptr);
}

/// Code for assessing tree quality

// Note: undefined if called on node without children
float surface_area_heuristic(const Cluster *root)
{
  // http://ompf2.com/viewtopic.php?f=3&t=206&start=10
  // HMC: t_cost and i_cost are not known
  // these can be empirically determined, but t_cost << i_cost
  // Note: Code copied from bvh_utils.cpp; for proper comparison be sure to
  // check that these values are the same for both!
  constexpr float32 t_cost = 1.0f;
  constexpr float32 i_cost = 1.0f;
  constexpr float32 primitives_per_leaf = 1;

  AABB<> aabb = root->aabb;
  float32 sa = surface_area(aabb);

  const Cluster *left_child = root->left;
  const Cluster *right_child = root->right;

  AABB<> left_aabb = left_child->aabb;
  AABB<> right_aabb = right_child->aabb;

  float32 left_sah, right_sah;
  if(left_child->isLeaf())
  {
    left_sah = (surface_area(left_aabb) / sa) * i_cost * primitives_per_leaf;
  }
  else
  {
    left_sah = surface_area_heuristic(left_child);
  }

  if(right_child->isLeaf())
  {
    right_sah = (surface_area(right_aabb) / sa) * i_cost * primitives_per_leaf;
  }
  else
  {
    right_sah = surface_area_heuristic(right_child);
  }

  return t_cost +
         (surface_area(left_aabb) / sa) * left_sah +
         (surface_area(right_aabb) / sa) * right_sah;

}

/// Debugging methods

// Returns the total size of the tree given a pointer to the root cluster
size_t getTreeSize(Cluster *root)
{
  size_t accumulator = 1;
  if (root -> left != nullptr)
    accumulator += getTreeSize(root->left);

  if (root -> right != nullptr)
    accumulator += getTreeSize(root->right);

  return accumulator;

}

// Returns a string representation of a given node comprising:
// its memory address, its cluster type, whether or not it is a leaf
// Cluster type:
//   0 -> root of tree
//   1 -> this node is the left child of its parent
//   2 -> this node is the right child of its parent
// isLeaf:
//   0 -> internal node
//   1 -> leaf node
//
// This method is used in printTree
std::string getName(const Cluster *node)
{
  long long address = (long long) node;
  return "\"" + std::to_string(address) + ", " + std::to_string(node->cluster_type) + ", " + std::to_string(node->isLeaf()) + "\"";

}

// Prints GraphViz-compatible representation of tree given pointer to root
void printTree(const Cluster *root)
{
  std::cout << getName(root) << "\n";

  if (root->cluster_type != Cluster::Root)
  {
    std::cout << "" << getName(root) << " -> " << getName(root->parent) << "\n";
  }
  if (!root->isLeaf())
  {
    std::cout << getName(root) << " -> " << getName(root->left) << "\n";
    std::cout << getName(root) << " -> " << getName(root->right) << "\n";

    printTree(root->left);
    printTree(root->right);
  }
}

// Prints GraphViz-compatible representation of forest given list of clusters
void printForest(const std::vector<Cluster *> &clusters)
{
  std::cout << "digraph G {\n";
  for (Cluster* v : clusters) {
    printTree(v);
  }
  std::cout << "}" << std::endl;
}

// Counts the leaves in a tree given pointer to root
size_t countLeaves(const Cluster *root)
{
  if (root->isLeaf())
    return 1;

  return countLeaves(root->left) + countLeaves(root->right);

}

// Counts the leaves in a forest given list of clusters
size_t countLeaves(const std::vector<Cluster *> &clusters)
{
  size_t accumulator = 0;
  for (Cluster *v : clusters) {
    accumulator += countLeaves(v);
  }

  return accumulator;
}

// Checks to see whether the AABBs of the children of the node fit within the
// AABB of the node given pointer to root
bool treeMakesSense(const Cluster *root)
{
  if (root->isLeaf())
    return true;

  const auto &AABB = root->aabb;
  const auto &leftAABB = root->left->aabb;
  const auto &rightAABB = root->right->aabb;

  return (AABB.contains(leftAABB) && AABB.contains(rightAABB)
      && treeMakesSense(root->left) && treeMakesSense(root->right));

}

// Checks to see whether the AABBs of the children of the node fit within the
// AABB of the node given flat_bvh representation
bool treeMakesSense(const Array<AABB<>> &aabbs, const Array<Vec<float32,4>> &flat_bvh)
{
  const AABB<> *aabbs_ptr = aabbs.get_device_ptr_const();
  const Vec<float32,4> *flat_bvh_ptr = flat_bvh.get_device_ptr_const();
  bool makesSense = true;
  for (size_t i = 0; i < flat_bvh.size()/4; ++i)
  {
    int left_index = reinterpret_cast<const int &>(flat_bvh_ptr[i*4 + 3][0]);
    int right_index = reinterpret_cast<const int &>(flat_bvh_ptr[i*4 + 3][1]);

    Vec<float32, 3> left_mins = {flat_bvh_ptr[i*4][0], flat_bvh_ptr[i*4][1], flat_bvh_ptr[i*4][2]};
    Vec<float32, 3> left_maxs = {flat_bvh_ptr[i*4][3], flat_bvh_ptr[i*4+1][0], flat_bvh_ptr[i*4+1][1]};

    Vec<float32, 3> right_mins = {flat_bvh_ptr[i*4+1][2], flat_bvh_ptr[i*4+1][3], flat_bvh_ptr[i*4+2][0]};
    Vec<float32, 3> right_maxs = {flat_bvh_ptr[i*4+2][1], flat_bvh_ptr[i*4+2][2], flat_bvh_ptr[i*4+2][3]};

    Vec<float32, 3> child_left_mins;
    Vec<float32, 3> child_left_maxs;

    Vec<float32, 3> child_right_mins;
    Vec<float32, 3> child_right_maxs;

    if (left_index < 0)
    {
      const auto &child_AABB = aabbs_ptr[-left_index - 1];
      child_left_mins = child_AABB.min();
      child_left_maxs = child_AABB.max();
    } else
    {
      child_left_mins = {
        std::min(flat_bvh_ptr[left_index][0], flat_bvh_ptr[left_index+1][2]),
        std::min(flat_bvh_ptr[left_index][1], flat_bvh_ptr[left_index+1][3]),
        std::min(flat_bvh_ptr[left_index][2], flat_bvh_ptr[left_index+2][0])
      };

      child_left_maxs = {
        std::max(flat_bvh_ptr[left_index][3], flat_bvh_ptr[left_index+2][1]),
        std::max(flat_bvh_ptr[left_index+1][0], flat_bvh_ptr[left_index+2][2]),
        std::max(flat_bvh_ptr[left_index+1][1], flat_bvh_ptr[left_index+2][3])
      };
    }

    if (right_index < 0)
    {
      const auto &child_AABB = aabbs_ptr[-right_index - 1];
      child_right_mins = child_AABB.min();
      child_right_maxs = child_AABB.max();
    } else
    {
      child_right_mins = {
        std::min(flat_bvh_ptr[right_index][0], flat_bvh_ptr[right_index+1][2]),
        std::min(flat_bvh_ptr[right_index][1], flat_bvh_ptr[right_index+1][3]),
        std::min(flat_bvh_ptr[right_index][2], flat_bvh_ptr[right_index+2][0])
      };

      child_right_maxs = {
        std::max(flat_bvh_ptr[right_index][3], flat_bvh_ptr[right_index+2][1]),
        std::max(flat_bvh_ptr[right_index+1][0], flat_bvh_ptr[right_index+2][2]),
        std::max(flat_bvh_ptr[right_index+1][1], flat_bvh_ptr[right_index+2][3])
      };
    }

    bool thisBoxOK = true;
    for (size_t j = 0; j < 3; ++j)
    {
      thisBoxOK &= left_mins[j] <= child_left_mins[j];
      thisBoxOK &= left_maxs[j] >= child_left_maxs[j];
      thisBoxOK &= right_mins[j] <= child_right_mins[j];
      thisBoxOK &= right_maxs[j] >= child_right_maxs[j];
    }

    makesSense &= thisBoxOK;

  }

  return makesSense;
}

/// End debugging functions

/// AAC functions

// Takes as input a list of Morton codes and the start and end indices defining
// the sublist to consider, as well as the partition bit
// Returns the index of the point about which to partition
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

// Takes as input the list of AABBs, the list of clusters, and the index i of a
// particular cluster
// Returns the index of the cluster closest to it
// Note: Given two clusters i and j, the closeness is defined as the surface
// area of the bounding box enclosing both i and j
size_t
findBestMatch(const Array<AABB<>> &aabbs, const std::vector<Cluster *> &clusters, size_t i)
{
  const AABB<> *aabbs_ptr = aabbs.get_device_ptr_const();

  float closestDist = std::numeric_limits<float>::infinity();
  size_t idx = i;
  for (size_t j = 0; j < clusters.size(); ++j) {
    if (i == j) continue;

    AABB<> combined = (clusters[i]->aabb).combine(clusters[j]->aabb);
    float d = surface_area(combined);
    if (d < closestDist) {
      closestDist = d;
      idx = j;
    }
  }

  return idx;
}

// Takes as input the list of AABBs, the list of clusters, the target number of
// clusters n, and a reference to the total number of nodes in the tree
// Returns nothing but modifies clusters and total_nodes in place
void
combineClusters(const Array<AABB<>> &aabbs, std::vector<Cluster *> &clusters, size_t n)
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
      const AABB<> combined = (clusters[i] -> aabb).combine(clusters[closest[i]] -> aabb);
      float d = surface_area(combined);
      if (d < best_dist)
      {
        best_dist = d;
        left_idx = i;
        right_idx = closest[i];
      }
    }

    // clusters at left_idx and right_idx are the best candidates for merging
    Cluster *left = clusters[left_idx];
    Cluster *right = clusters[right_idx];

    // combined is the smallest box enclosing both left and right
    // it is the box for the new cluster we will make
    AABB<> combined = left->aabb.combine(right->aabb);

    // Construct an internal node with the combined bounding box and the
    // appropriate child pointers
    Cluster *newNode = new Cluster(combined, left, right);

    // Update the pointers for the children so we can find the parent later
    left->parent = newNode;
    left->cluster_type = Cluster::LeftChild;

    right->parent = newNode;
    right->cluster_type = Cluster::RightChild;

    // we have combined two clusters into one, so now our list of clusters
    // can shrink by one

    // vectors have O(1) pop_back so we want to do some rearranging
    // newNode has left and right as children, so we want to do the following:
    //   newNode goes to where left used to be
    //   the last element in the list of clusters goes to where right used to be
    //   we then pop off the last node and |clusters| is now one less
    clusters[left_idx] = newNode;
    clusters[right_idx] = clusters.back();
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

// Takes as input the list of AABBs, the list of primitive IDs, the Morton codes
// of the AABBs, the start and end of the sublist to consider, the total nodes,
// and a partition bit (used for makePartition).
// Returns nothing but modifies clusters and total_nodes in place.
// Note: the clusters vector passed in should be empty; it is effectively a
// returned value
void
buildTree(const Array<AABB<>> &aabbs, const Array<int32> &primitive_ids,
    const Array<uint32> &mcodes, size_t start,
    size_t end, size_t partition_bit,
    std::vector<Cluster*> &clusters)
{

  if (end-start == 0)
  {
      return;
  }

  // base case
  // if the number of primitives coming in is less than delta
  if (end-start < AAC_DELTA)
  {
    const int32 *primitive_ids_ptr = primitive_ids.get_device_ptr_const();

    const AABB<> *aabbs_ptr = aabbs.get_device_ptr_const();

    // we want to initialize a bunch of primitive clusters
    clusters.resize(end-start);

    for (size_t i = start; i < end; ++i)
    {
        // Create primitive cluster
        Cluster *node = new Cluster(aabbs_ptr[i], i);

        clusters[i - start] = node;
    }

    combineClusters(aabbs, clusters, AAC_F(AAC_DELTA));
  } else
  {

    size_t splitIdx = makePartition(mcodes, start, end, partition_bit);

    size_t new_partition_bit = partition_bit - 1;
    std::vector<Cluster *> right_clusters;

    // clusters starts out as empty so we use it as left_clusters for now
    buildTree(aabbs, primitive_ids, mcodes, start, splitIdx, new_partition_bit, clusters);
    buildTree(aabbs, primitive_ids, mcodes, splitIdx, end, new_partition_bit, right_clusters);

    clusters.insert( clusters.end(), right_clusters.begin(), right_clusters.end() );

    combineClusters(aabbs, clusters, AAC_F(end-start));

  }
}

/// Devil Ray specific logistics

// Converts hierarchy defined by root_node into the proper flat_bvh structure
// expected by the rest of the codebase
Array<Vec<float32, 4>>
emit(const Array<AABB<>> &aabbs, Cluster *root_node)
{
  const int leaf_nodes = aabbs.size();
  const int inner_size = leaf_nodes - 1;

  Array<Vec<float32,4>> flat_bvh;
  flat_bvh.resize(inner_size * 4);
  Vec<float32,4> *flat_bvh_ptr = flat_bvh.get_device_ptr();

  // populate flat_bvh data structure using a work queue
  std::queue<Cluster *> todo;
  todo.push(root_node);

  // mapping between pointers to clusters we have already seen and their index
  // in the structure
  std::map<Cluster *, size_t> cluster_indices;

  size_t array_index = 0;

  // This is kind of weird code, here's a brief explanation:
  // Children fill in their parents' data entries.
  // Each node in our tree knows whether it is a left child or a right child,
  // and we have a mapping of Cluster pointers to their index in the array, so
  // a child will find its parent in the structure and fill in its bounding box
  // as well as its index.

  // While there is still work to do
  while (!todo.empty())
  {
    Cluster *this_node = todo.front();
    todo.pop();

    if (this_node->cluster_type != Cluster::Root)
    {
      // Here we find the parent so we can populate its data
      auto it = cluster_indices.find(this_node -> parent);

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
        current_index = -(this_node->aabb_id + 1);
      } else
      {
        // we need to keep track of this node's index so that its children
        // can populate it
        cluster_indices.insert({this_node, array_index});

        current_index = 4*array_index;
      }

      // parent_idx * 4 + 3 corresponds to the last Vec4 of floats
      // the 0th element is the left child index, the 1st element is the right
      // offset holds 0 for left, 1 for right from above
      flat_bvh_ptr[parent_idx * 4 + 3][offset] = reinterpret_cast<float32 &>(current_index);
    }

    if (!(this_node->isLeaf()))
    {
      // if we are at a leaf node, we don't care about holding on to its index
      // anymore, nor do we need to save space for it in the structure
      cluster_indices.insert({this_node, array_index});
      ++array_index;

      // and we definitely don't need to worry about its children
      todo.push(this_node->left);
      todo.push(this_node->right);
    }

  }

  return flat_bvh;
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

  // Build AAC
  std::vector<Cluster *> clusters;
  buildTree(aabbs, primitive_ids, mcodes, 0, aabbs.size(), 0, clusters);

  combineClusters(aabbs, clusters, 1);
  DRAY_LOG_ENTRY("build_tree", timer.elapsed());
  timer.reset();

  DRAY_LOG_ENTRY("sam", surface_area_heuristic(clusters[0]));

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

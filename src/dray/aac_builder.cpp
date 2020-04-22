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

namespace dray
{

static inline float AAC_C() {
    return (0.5f * powf(AAC_DELTA, 0.5f + AAC_EPSILON));
}

static inline uint32_t AAC_F(uint32_t x)
{
    return (uint32_t) (ceil(AAC_C() * powf(x, AAC_ALPHA)));
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

bool Cluster::isLeaf() const {
  return (this->left == nullptr && this->right == nullptr);
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

std::string getName(const Cluster *node)
{
  long long address = (long long) node;
  return "\"" + std::to_string(address) + ", " + std::to_string(node->cluster_type) + ", " + std::to_string(node->isLeaf()) + "\"";

}

// helper function for printing tree using graphviz
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

void printForest(const std::vector<Cluster *> &clusters)
{
  std::cout << "digraph G {\n";
  for (Cluster* v : clusters) {
    printTree(v);
  }
  std::cout << "}" << std::endl;
}

size_t countLeaves(const Cluster *root)
{
  if (root->isLeaf())
    return 1;

  return countLeaves(root->left) + countLeaves(root->right);

}

size_t countLeaves(const std::vector<Cluster *> &clusters)
{
  size_t accumulator = 0;
  for (Cluster *v : clusters) {
    accumulator += countLeaves(v);
  }

  return accumulator;
}

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

    AABB<> combined = (clusters[i]->aabb).onion(clusters[j]->aabb);
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

    clusters[left_idx]->parent = node;
    clusters[left_idx]->cluster_type = Cluster::LeftChild;

    clusters[right_idx]->parent = node;
    clusters[right_idx]->cluster_type = Cluster::RightChild;

    node->left = clusters[left_idx];
    node->right = clusters[right_idx];
    node->aabb = (node->left->aabb).onion(node->right->aabb);

    clusters[left_idx] = node;
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

// FIXME: document
// clusters is JUST the output
void
buildTree(const Array<AABB<>> &aabbs, const Array<int32> &primitive_ids,
    const Array<uint32> &mcodes, size_t start,
    size_t end, size_t &total_nodes, size_t partition_bit,
    std::vector<Cluster*> &clusters)
{

  if (end-start == 0)
  {
      return;
  }


  int dim = partition_bit % 3;

  if (end-start < AAC_DELTA)
  {
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
    }

    combineClusters(aabbs, clusters, AAC_F(AAC_DELTA), total_nodes, dim);

    return;
  }

  size_t splitIdx = makePartition(mcodes, start, end, partition_bit);

  size_t new_partition_bit = partition_bit - 1;
  std::vector<Cluster *> left_clusters;
  std::vector<Cluster *> right_clusters;
  size_t right_total_nodes = 0;

  buildTree(aabbs, primitive_ids, mcodes, start, splitIdx, total_nodes, new_partition_bit, left_clusters);
  buildTree(aabbs, primitive_ids, mcodes, splitIdx, end, right_total_nodes, new_partition_bit, right_clusters);

  total_nodes += right_total_nodes;

  left_clusters.insert( left_clusters.end(), right_clusters.begin(), right_clusters.end() );

  combineClusters(aabbs, left_clusters, AAC_F(end-start), total_nodes, dim);

  clusters = left_clusters;
}

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

    if (this_node->cluster_type != Cluster::Root)
    {
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
        // this is a leaf node
        current_index = -(this_node->aabb_id + 1);
      } else
      {
        // this is an internal node
        cluster_indices.insert({this_node, array_index});

        current_index = 4*array_index;
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
  size_t total_nodes = 0;
  buildTree(aabbs, primitive_ids, mcodes, 0, aabbs.size(),
      total_nodes, 0, clusters);

  combineClusters(aabbs, clusters, 1, total_nodes, 2);

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

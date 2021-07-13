// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/quadtree.hpp>
#include <dray/array_utils.hpp>

#include <conduit.hpp>
#include <conduit/conduit_relay.hpp>
#include <conduit/conduit_blueprint.hpp>

namespace dray
{

  // ====================================
  // QuadTreeForest
  // ====================================

  // Siblings are stored together.
  // Roots of the trees in the forest are considered siblings.
  // Nodes of the forest are numbered starting at 0.
  // Fields:
  //   - first_child: Each node stores the index of its first child.
  //   - parent: Each set of non-root siblings stores index of parent.
  //   - valid: Each set of non-root siblings stores bitfield of 'valid' bits.

  // QuadTreeForest(): deep copy constructor
  QuadTreeForest::QuadTreeForest(const QuadTreeForest & that)
    : m_num_trees(that.m_num_trees),
      m_num_nodes(that.m_num_nodes)
  {
    array_copy(m_first_child, that.m_first_child);
    array_copy(m_parent, that.m_parent);
    array_copy(m_valid, that.m_valid);
  }

  // QuadTreeForest(): move constructor
  QuadTreeForest::QuadTreeForest(QuadTreeForest && that)
    : m_num_trees(that.m_num_trees),
      m_num_nodes(that.m_num_nodes),
      m_first_child(that.m_first_child),
      m_parent(that.m_parent),
      m_valid(that.m_valid)
  {
  }

  // operator=() deep copy assignment
  const QuadTreeForest & QuadTreeForest::operator=(const QuadTreeForest &that)
  {
    m_num_trees = that.m_num_trees;
    m_num_nodes = that.m_num_nodes;
    array_copy(m_first_child, that.m_first_child);
    array_copy(m_parent, that.m_parent);
    array_copy(m_valid, that.m_valid);
    return *this;
  }

  // operator=() move assignment
  QuadTreeForest & QuadTreeForest::operator=(QuadTreeForest && that)
  {
    m_num_trees = that.m_num_trees;
    m_num_nodes = that.m_num_nodes;
    m_first_child = that.m_first_child;
    m_parent = that.m_parent;
    m_valid = that.m_valid;
    return *this;
  }


  // resize()
  void QuadTreeForest::resize(int32 num_trees)
  {
    m_num_trees = num_trees;
    m_num_nodes = num_trees;

    m_first_child.resize(num_trees);
    array_memset(m_first_child, -1);

    m_parent.resize(0);
    m_valid.resize(0);
  }

  // bytes_per_node()
  Float QuadTreeForest::bytes_per_node()
  {
    return Float(sizeof(TreeNodePtr))
           + Float(sizeof(TreeNodePtr)) / NUM_CHILDREN
           + Float(sizeof(QuadSiblingsValid)) / NUM_CHILDREN;
  }

  // num_trees()
  int32 QuadTreeForest::num_trees() const
  {
    return m_num_trees;
  }

  // num_nodes()
  int32 QuadTreeForest::num_nodes() const
  {
    return m_num_nodes;
  }

  // num_leafs()
  int32 QuadTreeForest::num_leafs() const
  {
    return m_leafs.get().size();
  }

  // num_valid_leafs()
  int32 QuadTreeForest::num_valid_leafs() const
  {
    ConstDeviceArray<TreeNodePtr> d_first_child(m_first_child);
    ConstDeviceArray<QuadSiblingsValid> d_valid(m_valid);
    const int32 num_nodes = this->num_nodes();
    const int32 num_trees = this->num_trees();
    RAJA::ReduceSum<reduce_policy, int32> count(0);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_nodes), [=] DRAY_LAMBDA (int32 n)
    {
      const bool is_leaf = d_first_child.get_item(n) < 0;
      const bool is_root = n < num_trees;

      bool counted = is_leaf;
      if (is_leaf && !is_root)
      {
        const uint8 valid_flag = d_valid.get_item((n - num_trees) / NUM_CHILDREN);
        const bool is_valid = (valid_flag >> ((n - num_trees) % NUM_CHILDREN)) & 1u;
        counted &= is_valid;
      }

      count += counted;
    });

    return count.get();
  }

  // leafs()
  Array<int32> QuadTreeForest::leafs() const
  {
    return m_leafs.get();
  }

  // capacity_nodes()
  int32 QuadTreeForest::capacity_nodes() const
  {
    return m_first_child.size();
  }

  // reserve_nodes()
  QuadTreeForest::ExpansionPlan
  QuadTreeForest::reserve_nodes(int32 new_cap) const
  {
    ExpansionPlan plan;
    const int32 old_cap = this->capacity_nodes();
    if (new_cap <= old_cap)
      new_cap = old_cap;
    else
      new_cap = max(new_cap, 2 * old_cap);

    plan.m_old_cap = old_cap;
    plan.m_new_cap = new_cap;
    return plan;
  }

  // execute_refinements()
  QuadTreeForest::ExpansionPlan
  QuadTreeForest::execute_refinements(Array<int32> nodal_flags)
  {
    // Make space for new leafs.
    // -------------------------
    ConstDeviceArray<int32> d_nodal_flags(nodal_flags);
    ConstDeviceArray<TreeNodePtr> d_leafs(this->leafs());
    const int32 old_leafs = this->num_leafs();
    const int32 old_nodes = this->num_nodes();

    Array<int32> refining = array_where_true(
        old_leafs,
        [=] DRAY_LAMBDA (int32 ii) {
          return d_nodal_flags.get_item(d_leafs.get_item(ii));
        });
    const int32 new_parents = refining.size();
    const int32 new_leafs = new_parents * NUM_CHILDREN;

    ExpansionPlan plan = this->reserve_nodes(old_nodes + new_leafs);
    this->execute_expansion(plan);

    // Add new leafs to the tree.
    // --------------------------
    ConstDeviceArray<int32> d_refining(refining);
    NonConstDeviceArray<TreeNodePtr> d_first_child(m_first_child);
    NonConstDeviceArray<TreeNodePtr> d_parent(m_parent);
    NonConstDeviceArray<QuadSiblingsValid> d_valid(m_valid);
    const int32 num_trees = this->num_trees();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, new_parents),
        [=] DRAY_LAMBDA (int32 i)
    {
      const TreeNodePtr family = old_nodes + i * NUM_CHILDREN;
      const TreeNodePtr parent = d_leafs.get_item(d_refining.get_item(i));
      d_first_child.get_item(parent) = family;
      for (int32 child = 0; child < NUM_CHILDREN; ++child)
        d_first_child.get_item(family + child) = -1;
      d_parent.get_item((family - num_trees) / NUM_CHILDREN) = parent;
      d_valid.get_item((family - num_trees) / NUM_CHILDREN)
          = (1u << NUM_CHILDREN) - 1;  // all children valid
    });
    m_num_nodes += new_leafs;

    this->m_leafs.reset();
    return plan;
  }

  // execute_expansion()
  void QuadTreeForest::execute_expansion(
      const ExpansionPlan &plan)
  {
    const int32 num_roots = this->num_trees();
    const int32 num_nodes = this->num_nodes();

    m_first_child = array_resize_copy(m_first_child, plan.m_new_cap, -1);
    m_parent = array_resize_copy(
        m_parent, (plan.m_new_cap - num_roots) / NUM_CHILDREN, -1);
    m_valid = array_resize_copy(
        m_valid, (plan.m_new_cap - num_roots) / NUM_CHILDREN, uint8(0));
  }

  // IndexLeafs recipe
  Array<TreeNodePtr> QuadTreeForest::IndexLeafs::operator()
    (const QuadTreeForest *forest) const
  {
    ConstDeviceArray<TreeNodePtr> d_first_child(forest->m_first_child);
    return array_where_true(forest->num_nodes(),
        [=] DRAY_LAMBDA (int32 ii) { return d_first_child.get_item(ii) < 0; });
  }


  // ====================================
  // DeviceQuadTreeForest
  // ====================================

    DeviceQuadTreeForest::DeviceQuadTreeForest(const QuadTreeForest &host)
      : m_num_trees(host.m_num_trees),
        m_num_nodes(host.m_num_nodes),
        m_num_leafs(host.num_leafs()),
        m_first_child(host.m_first_child),
        m_parent(host.m_parent),
        m_valid(host.m_valid)
    { }


}//namespace dray

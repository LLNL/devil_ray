// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/quadtree.hpp>
#include <dray/array_utils.hpp>

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
      d_parent.get_item((family - num_trees) / NUM_CHILDREN) = parent;
      d_valid.get_item((family - num_trees) / NUM_CHILDREN)
          = (1u << NUM_CHILDREN) - 1;  // all children valid
    });
    m_num_nodes += new_leafs;

    this->m_leafs.reset();
    return plan;
  }

  // expand_node_array()
  template <typename T>
  void QuadTreeForest::expand_node_array(
      const ExpansionPlan &plan, Array<T> &a) const
  {
    a = array_resize_copy(a, plan.m_new_cap);
  }

  // execute_expansion()
  void QuadTreeForest::execute_expansion(
      const ExpansionPlan &plan)
  {
    const int32 num_roots = this->num_trees();
    const int32 num_nodes = this->num_nodes();

    m_first_child = array_resize_copy(m_first_child, plan.m_new_cap);
    m_parent = array_resize_copy(
        m_parent, (plan.m_new_cap - num_roots) / NUM_CHILDREN);
    m_valid = array_resize_copy(
        m_valid, (plan.m_new_cap - num_roots) / NUM_CHILDREN);
  }

  // IndexLeafs recipe
  Array<TreeNodePtr> QuadTreeForest::IndexLeafs::operator()
    (const QuadTreeForest *forest) const
  {
    ConstDeviceArray<TreeNodePtr> d_first_child(forest->m_first_child);
    return array_where_true(forest->num_nodes(),
        [=] DRAY_LAMBDA (int32 ii) { return d_first_child.get_item(ii) < 0; });
  }

  // integrate_phys_area_to_mesh()
  template <class DeviceLocationToJacobian,
           class DeviceFaceLocationToScalar>
  IntegrateToMesh QuadTreeForest::integrate_phys_area_to_mesh(
      Array<FaceLocation> face_centers,
      const DeviceLocationToJacobian &phi_prime,
      const DeviceFaceLocationToScalar &integrand)
  {
    IntegrateToMesh integrate_to_mesh;

    const int32 num_leafs = this->num_leafs();
    ConstDeviceArray<TreeNodePtr> d_leafs(this->m_leafs.get());
    ConstDeviceArray<FaceLocation> d_face_centers(face_centers);
    DeviceQuadTreeForest d_forest(*this);

    RAJA::ReduceSum<reduce_policy, Float> sum(0.0f);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_leafs),
        [=] DRAY_LAMBDA (int32 ii)
    {
      // Quadrant q
      TreeNodePtr leaf_node = d_leafs.get_item(ii);
      QuadTreeQuadrant logical_q = d_forest.quadrant(leaf_node);
      FaceLocation face_center =
        d_face_centers.get_item(logical_q.tree_id());
      Quadrant q = Quadrant::create(face_center, logical_q);

      // Evaluate physical-space area of the (sub)quad.
      constexpr int32 ncomp = 3;
      Vec<Vec<Float, ncomp>, 3> vol_jacobian = phi_prime(q.center().loc());
      Vec<Vec<Float, ncomp>, 2> face_jacobian;
      q.world_tangents(vol_jacobian, face_jacobian[0], face_jacobian[1]);
      Float dA = cross(face_jacobian[0], face_jacobian[1]).magnitude();

      // Integrate (trapezoidal rule).
      Float region_value = 0.0f;
      region_value += integrand(q.lower_left());
      region_value += integrand(q.lower_right());
      region_value += integrand(q.upper_left());
      region_value += integrand(q.upper_right());
      region_value /= 4;
      
      sum += region_value * dA;
    });

    integrate_to_mesh.m_result = sum.get();
    return integrate_to_mesh;
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

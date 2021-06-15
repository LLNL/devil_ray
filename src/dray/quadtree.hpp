// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_QUADTREE_HPP
#define DRAY_QUADTREE_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/location.hpp>
#include <dray/face_location.hpp>
#include <dray/lazy_prop.hpp>
#include <dray/array.hpp>
#include <dray/device_array.hpp>
#include <dray/integrate.hpp>
#include <dray/data_model/mesh.hpp>

namespace dray
{
  /** Axis-aligned reference-space square in a mesh. */
  struct Quadrant;

  /** Adaptive quad-tree on [0,1]x[0,1] */
  struct QuadTreeForest;
  struct DeviceQuadTreeForest;
  struct QuadTreeQuadrant;

  // ----------------------------------- //

  struct Quadrant
  {
    FaceLocation m_loc;  // center
    Float m_side;

    DRAY_EXEC static Quadrant create(
        const FaceLocation &face_center, const QuadTreeQuadrant &q);

    DRAY_EXEC FaceLocation center() const;
    DRAY_EXEC FaceLocation upper_right() const;
    DRAY_EXEC FaceLocation upper_left() const;
    DRAY_EXEC FaceLocation lower_right() const;
    DRAY_EXEC FaceLocation lower_left() const;

    // Partial derivatives along a face.
    // Useful for area: physical dA = |cross(t0, t1)|.
    template <int32 ncomp>
    DRAY_EXEC void world_tangents(const Vec<Vec<Float, ncomp>, 3> &jacobian,
                                  Vec<Float, ncomp> &t0,
                                  Vec<Float, ncomp> &t1) const;

    DRAY_EXEC Float world_area(const Vec<Vec<Float, 3>, 3> &jacobian) const;
  };


  // ====================================
  typedef int32 TreeNodePtr;
  typedef uint8 QuadSiblingsValid;
  // ====================================

  // ====================================
  // QuadTreeForest
  // ====================================
  struct QuadTreeForest
  {
    // (Note that modifying QuadTreeForest after
    // creating a DeviceQuadTreeForest view may
    // invalidate the view.)
    friend struct DeviceQuadTreeForest;

    void resize(int32 num_trees);

    static Float bytes_per_node();
    int32 num_trees() const;
    int32 num_nodes() const;
    int32 num_leafs() const;
    int32 num_valid_leafs() const;
    Array<TreeNodePtr> leafs() const;

    struct ExpansionPlan
    {
      int32 m_old_cap;
      int32 m_new_cap;
    };

    int32 capacity_nodes() const;
    ExpansionPlan reserve_nodes(int32 new_cap) const;
    ExpansionPlan execute_refinements(Array<int32> nodal_flags);

    template <typename T>
    void expand_node_array(const ExpansionPlan &plan,
                           Array<T> &a) const;
    template <typename T>
    void expand_node_array(const ExpansionPlan &plan,
                           Array<T> &a,
                           const T &fillv) const;
    void execute_expansion(const ExpansionPlan &plan);

    template <class DeviceLocationToJacobian,
             class DeviceFaceLocationToScalar>
    IntegrateToMesh integrate_phys_area_to_mesh(
        Array<FaceLocation> face_centers,
        const DeviceLocationToJacobian &phi_prime,
        const DeviceFaceLocationToScalar &integrand);

    //-------------------------
    static constexpr int32 NUM_CHILDREN = 4;
    static constexpr int32 NUM_DIMS = 2;
    int32 m_num_trees;
    int32 m_num_nodes;
    Array<TreeNodePtr> m_first_child;
    Array<TreeNodePtr> m_parent;
    Array<QuadSiblingsValid> m_valid;

    protected:
      // Indexing of leafs is derived from nodes (ground truth).
      struct IndexLeafs
      {
        Array<TreeNodePtr> operator()(const QuadTreeForest *) const;
      };

      LazyProp<Array<TreeNodePtr>, IndexLeafs, const QuadTreeForest *> m_leafs =
          {IndexLeafs(), this};
  };

  // ====================================
  // DeviceQuadTreeForest
  // ====================================
  struct DeviceQuadTreeForest
  {
    DeviceQuadTreeForest(const QuadTreeForest &);
    DeviceQuadTreeForest(const DeviceQuadTreeForest &) = default;
    DRAY_EXEC int32 num_trees() const;
    DRAY_EXEC int32 num_nodes() const;
    DRAY_EXEC int32 num_leafs() const;
    /// DRAY_EXEC int32 capacity_nodes() const;
    /// DRAY_EXEC int32 capacity_depth() const;
    // All nodes
    DRAY_EXEC bool valid(TreeNodePtr node) const;
    DRAY_EXEC bool leaf(TreeNodePtr node) const;
    DRAY_EXEC bool root(TreeNodePtr node) const;
    DRAY_EXEC TreeNodePtr child(TreeNodePtr node, int32 child_num) const;
    DRAY_EXEC TreeNodePtr parent(TreeNodePtr node) const;
    DRAY_EXEC int32 tree_id(TreeNodePtr node) const;
    DRAY_EXEC QuadTreeQuadrant quadrant(TreeNodePtr node) const;
    DRAY_EXEC static void child_quadrant(
        QuadTreeQuadrant &quadrant, int32 child_num);  // in place
    // Nonroot nodes
    DRAY_EXEC int32 child_num(TreeNodePtr node) const;
    // Nodes with a grandparent
    //-------------------------
    static constexpr int32 NUM_CHILDREN = 4;
    static constexpr int32 NUM_DIMS = 2;
    int32 m_num_trees;
    int32 m_num_nodes;
    int32 m_num_leafs;
    ConstDeviceArray<TreeNodePtr> m_first_child;
    ConstDeviceArray<TreeNodePtr> m_parent;
    ConstDeviceArray<QuadSiblingsValid> m_valid;

    private:
      DRAY_EXEC bool valid_bit(TreeNodePtr node) const;
  };


  // ====================================
  // QuadTreeQuadrant
  // ====================================
  struct QuadTreeQuadrant
  {
    int32 m_tree_id;
    int32 m_depth;
    Vec<Float, 2> m_center;

    DRAY_EXEC Float side() const;
    DRAY_EXEC int32 tree_id() const;
    DRAY_EXEC int32 depth() const;
    DRAY_EXEC Vec<Float, 2> center() const;
  };

}//namespace dray



namespace dray
{

  // ====================================
  // Quadrant
  // ====================================

  // center()
  DRAY_EXEC FaceLocation Quadrant::center() const
  {
    return m_loc;
  }

  // upper_right()
  DRAY_EXEC FaceLocation Quadrant::upper_right() const
  {
    FaceLocation loc = m_loc;
    loc.m_loc.m_ref_pt += loc.tangents().m_t[0].vec<Float, 3>() * 0.5 * m_side;
    loc.m_loc.m_ref_pt += loc.tangents().m_t[1].vec<Float, 3>() * 0.5 * m_side;
    return loc;
  }

  // upper_left()
  DRAY_EXEC FaceLocation Quadrant::upper_left() const
  {
    FaceLocation loc = m_loc;
    loc.m_loc.m_ref_pt -= loc.tangents().m_t[0].vec<Float, 3>() * 0.5 * m_side;
    loc.m_loc.m_ref_pt += loc.tangents().m_t[1].vec<Float, 3>() * 0.5 * m_side;
    return loc;
  }

  // lower_right()
  DRAY_EXEC FaceLocation Quadrant::lower_right() const
  {
    FaceLocation loc = m_loc;
    loc.m_loc.m_ref_pt += loc.tangents().m_t[0].vec<Float, 3>() * 0.5 * m_side;
    loc.m_loc.m_ref_pt -= loc.tangents().m_t[1].vec<Float, 3>() * 0.5 * m_side;
    return loc;
  }

  // lower_left()
  DRAY_EXEC FaceLocation Quadrant::lower_left() const
  {
    FaceLocation loc = m_loc;
    loc.m_loc.m_ref_pt -= loc.tangents().m_t[0].vec<Float, 3>() * 0.5 * m_side;
    loc.m_loc.m_ref_pt -= loc.tangents().m_t[1].vec<Float, 3>() * 0.5 * m_side;
    return loc;
  }

  // world_tangents()
  template <int32 ncomp>
  DRAY_EXEC void Quadrant::world_tangents(
      const Vec<Vec<Float, ncomp>, 3> &jacobian,
      Vec<Float, ncomp> &t0,
      Vec<Float, ncomp> &t1) const
  {
    m_loc.world_tangents(jacobian, t0, t1);
    t0 *= m_side;
    t1 *= m_side;
  }

  // world_area()
  DRAY_EXEC Float Quadrant::world_area(const Vec<Vec<Float, 3>, 3> &jacobian) const
  {
    Vec<Vec<Float, 3>, 2> face_jacobian;
    world_tangents(jacobian, face_jacobian[0], face_jacobian[1]);
    Float dA = cross(face_jacobian[0], face_jacobian[1]).magnitude();
    return dA;
  }

  // create()
  DRAY_EXEC Quadrant Quadrant::create(
      const FaceLocation &face_center, const QuadTreeQuadrant &lq)
  {
    const Vec<Float, 3> t[2] = {face_center.tangents().m_t[0].vec<Float, 3>(),
                                face_center.tangents().m_t[1].vec<Float, 3>()};
    const Vec<Float, 3> origin = face_center.m_loc.m_ref_pt - (t[0] * 0.5) - (t[1] * 0.5);

    Quadrant q;
    q.m_loc = face_center;
    q.m_loc.m_loc.m_ref_pt = origin + (t[0] * lq.center()[0]) + (t[1] * lq.center()[1]);
    q.m_side = lq.side();

    return q;
  }

  // ====================================
  // DeviceQuadTreeForest
  // ====================================

  // num_trees()
  DRAY_EXEC int32 DeviceQuadTreeForest::num_trees() const
  {
    return m_num_trees;
  }

  // num_nodes()
  DRAY_EXEC int32 DeviceQuadTreeForest::num_nodes() const
  {
    return m_num_nodes;
  }

  // num_leafs()
  DRAY_EXEC int32 DeviceQuadTreeForest::num_leafs() const
  {
    return m_num_leafs;
  }

  /// // capacity_nodes()
  /// DRAY_EXEC int32 DeviceQuadTreeForest::capacity_nodes() const
  /// {
  ///   return m_first_child.size();
  /// }

  /// // capacity_depth()
  /// DRAY_EXEC int32 DeviceQuadTreeForest::capacity_depth() const
  /// {
  ///   const int32 unused = capacity_nodes() - num_nodes();
  ///   const int32 per_leaf = unused / num_leafs();
  ///   const int32 depth = int32(log2(per_leaf)) / NUM_DIMS;
  ///   return depth;
  /// }

  // valid()
  DRAY_EXEC bool DeviceQuadTreeForest::valid(TreeNodePtr node) const
  {
    const bool in_bounds = node >= 0 && node < num_nodes();
    return in_bounds && (root(node) || valid_bit(node));
  }

  // valid_bit()
  DRAY_EXEC bool DeviceQuadTreeForest::valid_bit(TreeNodePtr node) const
  {
    const QuadSiblingsValid bitset =
      m_valid.get_item((node - m_num_trees)/NUM_CHILDREN);
    return (bitset >> child_num(node)) & 1u;
  }

  // leaf()
  DRAY_EXEC bool DeviceQuadTreeForest::leaf(TreeNodePtr node) const
  {
    return m_first_child.get_item(node) < 0;
  }

  // root()
  DRAY_EXEC bool DeviceQuadTreeForest::root(TreeNodePtr node) const
  {
    return (node < m_num_trees);
  }

  // child()
  DRAY_EXEC TreeNodePtr DeviceQuadTreeForest::child(
      TreeNodePtr node, int32 child_num) const
  {
    int32 first_child = m_first_child.get_item(node);
    return (first_child >= 0 ? first_child + child_num : -1);
  }

  // parent()
  DRAY_EXEC TreeNodePtr DeviceQuadTreeForest::parent(TreeNodePtr node) const
  {
    return (node < m_num_trees ? -1 :
        m_parent.get_item((node - m_num_trees)/NUM_CHILDREN));
  }

  // child_num()
  DRAY_EXEC int32 DeviceQuadTreeForest::child_num(TreeNodePtr node) const
  {
    return (node - m_num_trees) % NUM_CHILDREN;
  }

  // tree_id()
  DRAY_EXEC int32 DeviceQuadTreeForest::tree_id(TreeNodePtr node) const
  {
    while (!root(node))
      node = parent(node);
    return node;
  }

  // quadrant()
  DRAY_EXEC QuadTreeQuadrant DeviceQuadTreeForest::quadrant(
      TreeNodePtr node) const
  {
    QuadTreeQuadrant q;
    q.m_depth = 0;
    q.m_center = Vec<Float, 2>{{.5, .5}};
    while (!root(node))
    {
      q.m_depth++;

      const int32 cnum = child_num(node);
      q.m_center[0] += (cnum >> 0) & 1u;
      q.m_center[1] += (cnum >> 1) & 1u;
      q.m_center *= .5;

      node = parent(node);
    }
    q.m_tree_id = node;
    return q;
  }

  // child_quadrant()
  DRAY_EXEC void DeviceQuadTreeForest::child_quadrant(
      QuadTreeQuadrant &quadrant, int32 child_num)
  {
    quadrant.m_depth += 1;
    const Float scale = 1.0 / (1u << quadrant.m_depth);
    quadrant.m_center[0] += (((child_num >> 0) & 1u) - 0.5) * scale;
    quadrant.m_center[1] += (((child_num >> 1) & 1u) - 0.5) * scale;
    // quadrant.m_tree_id is the same because child is in the same tree.
  }

  // ====================================
  // QuadTreeQuadrant
  // ====================================

  // side()
  DRAY_EXEC Float QuadTreeQuadrant::side() const
  {
    return 1.0 / (1 << m_depth);
  }

  // tree_id()
  DRAY_EXEC int32 QuadTreeQuadrant::tree_id() const
  {
    return m_tree_id;
  }

  // depth()
  DRAY_EXEC int32 QuadTreeQuadrant::depth() const
  {
    return m_depth;
  }

  // center()
  DRAY_EXEC Vec<Float, 2> QuadTreeQuadrant::center() const
  {
    return m_center;
  }
}

#endif//DRAY_QUADTREE_HPP

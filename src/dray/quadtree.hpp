// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_QUADTREE_HPP
#define DRAY_QUADTREE_HPP

#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/location.hpp>
#include <dray/face_location.hpp>
#include <dray/lazy_prop.hpp>
#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/device_array.hpp>
#include <dray/integrate.hpp>
#include <dray/data_model/mesh.hpp>

#include <RAJA/RAJA.hpp>
#include <conduit_blueprint.hpp>

namespace conduit
{
  class Node;
}

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

    QuadTreeForest() = default;
    QuadTreeForest(const QuadTreeForest &);                    // deep copy
    QuadTreeForest(QuadTreeForest &&);                         // move
    const QuadTreeForest & operator=(const QuadTreeForest &);  // deep copy
    QuadTreeForest & operator=(QuadTreeForest &&);             // move

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

    // integrate_phys_area_to_mesh()
    template <class DeviceLocationToJacobian,
             class DeviceFaceLocationToScalar>
    IntegrateToMesh integrate_phys_area_to_mesh(
        Array<FaceLocation> face_centers,
        const DeviceLocationToJacobian &phi_prime,
        const DeviceFaceLocationToScalar &integrand);

    // to_blueprint()
    template <class DeviceLocationToXYZ,
              class DeviceFaceLocationToScalar>
    void
    to_blueprint(
        Array<FaceLocation> face_centers,
        const DeviceLocationToXYZ &phi,
        const DeviceFaceLocationToScalar &integrand,
        conduit::Node &n_dataset) const;

    // reference_tiles_to_blueprint()
    template <class DeviceFaceLocationToScalar>
    void
    reference_tiles_to_blueprint(
        Array<int32> faces_for_tiles,
        Array<Vec<Float, 2>> origins_of_tiles,
        Array<FaceLocation> face_centers,
        const DeviceFaceLocationToScalar &integrand,
        conduit::Node &n_dataset) const;

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
  // QuadTreeForest
  // ====================================

  // expand_node_array()
  template <typename T>
  void QuadTreeForest::expand_node_array(
      const ExpansionPlan &plan, Array<T> &a) const
  {
    a = array_resize_copy(a, plan.m_new_cap);
  }

  // expand_node_array()
  template <typename T>
  void QuadTreeForest::expand_node_array(
      const ExpansionPlan &plan, Array<T> &a, const T &fillv) const
  {
    a = array_resize_copy(a, plan.m_new_cap, fillv);
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

    RAJA::ReduceSum<reduce_policy, IntegrateT> sum(0.0f);

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
      Float dA = q.world_area(phi_prime(q.center().loc()));

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

  namespace detail
  {
    template <int32 ncomp>
    void set_external_vector_component_host(
        conduit::Node &node,
        Array<Vec<Float, ncomp>> &arr,
        int32 component)
    {
      Vec<Float, ncomp> * host_ptr = arr.get_host_ptr();
      node.set_external(
          (Float*) arr.get_host_ptr(),
          arr.size(),
          (uint8*)(&host_ptr[0][component]) - (uint8*)(host_ptr),
          (uint8*)(&host_ptr[1][component]) - (uint8*)(&host_ptr[0][component]));
    }
  }

  // to_blueprint()
  template <class DeviceLocationToXYZ,
            class DeviceFaceLocationToScalar>
  void
  QuadTreeForest::to_blueprint(
      Array<FaceLocation> face_centers,
      const DeviceLocationToXYZ &phi,
      const DeviceFaceLocationToScalar &integrand,
      conduit::Node &n_dataset) const
  {
    using namespace conduit;

    // Gather data needed for blueprint arrays.
    const int32 num_leafs = this->num_leafs();
    const int32 num_verts = num_leafs * 4;
    Array<Vec<Float, 3>> verts;
    Array<int32> conn;
    Array<Float> avg_value;
    verts.resize(num_verts);
    conn.resize(num_verts);
    avg_value.resize(num_leafs);
    //
    DeviceQuadTreeForest d_forest(*this);
    ConstDeviceArray<TreeNodePtr> d_leafs(this->leafs());
    ConstDeviceArray<FaceLocation> d_face_centers(face_centers);
    NonConstDeviceArray<Vec<Float, 3>> d_verts(verts);
    NonConstDeviceArray<int32> d_conn(conn);
    NonConstDeviceArray<Float> d_avg_value(avg_value);
    //
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_leafs), [=] DRAY_LAMBDA (int32 i)
    {
      // Get quadrant.
      const TreeNodePtr leaf = d_leafs.get_item(i);
      QuadTreeQuadrant logical_q = d_forest.quadrant(leaf);
      FaceLocation face_center =
        d_face_centers.get_item(logical_q.tree_id());
      Quadrant q = Quadrant::create(face_center, logical_q);

      // Vertices.
      d_verts.get_item(i * 4 + 0) = phi(q.lower_left().loc());
      d_verts.get_item(i * 4 + 1) = phi(q.lower_right().loc());
      d_verts.get_item(i * 4 + 2) = phi(q.upper_left().loc());
      d_verts.get_item(i * 4 + 3) = phi(q.upper_right().loc());

      // Connectivity.
      d_conn.get_item(i * 4 + 0) = i * 4 + 0;
      d_conn.get_item(i * 4 + 1) = i * 4 + 2;
      d_conn.get_item(i * 4 + 2) = i * 4 + 3;
      d_conn.get_item(i * 4 + 3) = i * 4 + 1;

      // Average value (trapezoidal rule).
      Float region_value = 0.0f;
      region_value += integrand(q.lower_left());
      region_value += integrand(q.lower_right());
      region_value += integrand(q.upper_left());
      region_value += integrand(q.upper_right());
      region_value /= 4;
      d_avg_value.get_item(i) = region_value;
    });

    // reset
    n_dataset.reset();

    // create the coordinate set
    Node &coordset = n_dataset["coordsets/coords"];
    coordset["type"] = "explicit";
    detail::set_external_vector_component_host(coordset["values/x"], verts, 0);
    detail::set_external_vector_component_host(coordset["values/y"], verts, 1);
    detail::set_external_vector_component_host(coordset["values/z"], verts, 2);

    // add the topology
    Node &topo = n_dataset["topologies/topo"];
    topo["type"] = "unstructured";
    topo["coordset"] = "coords";
    topo["elements/shape"] = "quad";
    topo["elements/connectivity"].set_external(conn.get_host_ptr(), conn.size());

    // add an element-associated field with the integrand values.
    Node &n_field = n_dataset["fields/avg_value"];
    n_field["association"] = "element";
    n_field["topology"] = "topo";
    n_field["values"].set_external(avg_value.get_host_ptr(), avg_value.size());

    // make sure we conform:
    Node verify_info;
    if(!blueprint::mesh::verify(n_dataset, verify_info))
    {
        std::cout << "Verify failed!" << std::endl;
        verify_info.print();
    }

    /// // print out results
    /// n_dataset.print();
  }


  // reference_tiles_to_blueprint()
  template <class DeviceFaceLocationToScalar>
  void
  QuadTreeForest::reference_tiles_to_blueprint(
      Array<int32> faces_for_tiles,
      Array<Vec<Float, 2>> origins_of_tiles,
      Array<FaceLocation> face_centers,
      const DeviceFaceLocationToScalar &integrand,
      conduit::Node &n_dataset) const
  {
    using namespace conduit;
    DeviceQuadTreeForest d_forest(*this);

    //
    // Select leafs within the chosen faces.
    //

    // Flag trees.
    Array<uint8> tree_selected;
    Array<int32> tree_to_selection_idx;
    tree_selected.resize(this->num_trees());
    tree_to_selection_idx.resize(this->num_trees());
    array_memset_zero(tree_selected);
    array_memset_zero(tree_to_selection_idx);
    ConstDeviceArray<int32> d_faces_for_tiles(faces_for_tiles);
    NonConstDeviceArray<uint8> d_tree_selected(tree_selected);
    NonConstDeviceArray<int32> d_tree_to_selection_idx(tree_to_selection_idx);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, faces_for_tiles.size()),
        [=] DRAY_LAMBDA (int32 i)
    {
      const TreeNodePtr tree = d_faces_for_tiles.get_item(i);
      d_tree_selected.get_item(tree) = true;
      d_tree_to_selection_idx.get_item(tree) = i;
    });

    // Select leafs.
    Array<TreeNodePtr> leafs = this->leafs();
    leafs = compact(leafs, [=] DRAY_LAMBDA (TreeNodePtr leaf)
            { return d_tree_selected.get_item(d_forest.tree_id(leaf)); });

    // Gather data needed for blueprint arrays.
    const int32 num_leafs = leafs.size();
    const int32 num_verts = num_leafs * 4;
    Array<Vec<Float, 3>> verts;
    Array<int32> conn;
    Array<Float> avg_value;
    verts.resize(num_verts);
    conn.resize(num_verts);
    avg_value.resize(num_leafs);
    //
    ConstDeviceArray<TreeNodePtr> d_leafs(leafs);
    ConstDeviceArray<Vec<Float, 2>> d_origins_of_tiles(origins_of_tiles);
    ConstDeviceArray<FaceLocation> d_face_centers(face_centers);
    NonConstDeviceArray<Vec<Float, 3>> d_verts(verts);
    NonConstDeviceArray<int32> d_conn(conn);
    NonConstDeviceArray<Float> d_avg_value(avg_value);
    //
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_leafs), [=] DRAY_LAMBDA (int32 i)
    {
      // Get quadrant.
      const TreeNodePtr leaf = d_leafs.get_item(i);
      const QuadTreeQuadrant logical_q = d_forest.quadrant(leaf);
      FaceLocation face_center =
        d_face_centers.get_item(logical_q.tree_id());
      Quadrant q = Quadrant::create(face_center, logical_q);

      // Get output tile.
      const int32 selection_idx = d_tree_to_selection_idx.get_item(logical_q.tree_id());
      const Vec<Float, 2> tile_origin = d_origins_of_tiles.get_item(selection_idx);

      // Face-relative vertices.
      Vec<Float, 2> face_coords[4];
      const FaceTangents tans = face_center.tangents();
      face_coords[0] = tans.project_ref_to_face(q.lower_left().loc().m_ref_pt);
      face_coords[1] = tans.project_ref_to_face(q.lower_right().loc().m_ref_pt);
      face_coords[2] = tans.project_ref_to_face(q.upper_left().loc().m_ref_pt);
      face_coords[3] = tans.project_ref_to_face(q.upper_right().loc().m_ref_pt);

      // 3D planar vertices in tiled output.
      const Vec<Float, 3> origin_3d = {{tile_origin[0], tile_origin[1], 0}};
      for (int32 d = 0; d < 4; ++d)
      {
        const Vec<Float, 2> vertex_2d = tile_origin + face_coords[d];
        const Vec<Float, 3> vertex_3d = {{vertex_2d[0], vertex_2d[1], 0}};
        d_verts.get_item(i * 4 + d) = vertex_3d;
      }

      // Connectivity.
      d_conn.get_item(i * 4 + 0) = i * 4 + 0;
      d_conn.get_item(i * 4 + 1) = i * 4 + 2;
      d_conn.get_item(i * 4 + 2) = i * 4 + 3;
      d_conn.get_item(i * 4 + 3) = i * 4 + 1;

      // Average value (trapezoidal rule).
      Float region_value = 0.0f;
      region_value += integrand(q.lower_left());
      region_value += integrand(q.lower_right());
      region_value += integrand(q.upper_left());
      region_value += integrand(q.upper_right());
      region_value /= 4;
      d_avg_value.get_item(i) = region_value;
    });

    // reset
    n_dataset.reset();

    // create the coordinate set
    Node &coordset = n_dataset["coordsets/coords"];
    coordset["type"] = "explicit";
    detail::set_external_vector_component_host(coordset["values/x"], verts, 0);
    detail::set_external_vector_component_host(coordset["values/y"], verts, 1);
    detail::set_external_vector_component_host(coordset["values/z"], verts, 2);

    // add the topology
    Node &topo = n_dataset["topologies/topo"];
    topo["type"] = "unstructured";
    topo["coordset"] = "coords";
    topo["elements/shape"] = "quad";
    topo["elements/connectivity"].set_external(conn.get_host_ptr(), conn.size());

    // add an element-associated field with the integrand values.
    Node &n_field = n_dataset["fields/avg_value"];
    n_field["association"] = "element";
    n_field["topology"] = "topo";
    n_field["values"].set_external(avg_value.get_host_ptr(), avg_value.size());

    // make sure we conform:
    Node verify_info;
    if(!blueprint::mesh::verify(n_dataset, verify_info))
    {
        std::cout << "Verify failed!" << std::endl;
        verify_info.print();
    }

    /// // print out results
    /// n_dataset.print();
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

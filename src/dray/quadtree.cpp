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

    DeviceQuadTreeForest::DeviceQuadTreeForest(const QuadTreeForest &host)
      : m_num_trees(host.m_num_trees),
        m_num_nodes(host.m_num_nodes),
        m_num_leafs(host.num_leafs()),
        m_first_child(host.m_first_child),
        m_parent(host.m_parent),
        m_valid(host.m_valid)
    { }


}//namespace dray

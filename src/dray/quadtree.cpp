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

    m_first_child.resize(num_trees);
    array_memset(m_first_child, -1);

    m_parent.resize(0);
    m_valid.resize(0);
  }

  // num_nodes()
  int32 QuadTreeForest::num_nodes() const
  {
    return m_first_child.size();
  }

  // num_leafs()
  int32 QuadTreeForest::num_leafs() const
  {
    return m_leafs.get().size();
  }

  // reserve_nonroot_nodes()
  /// void QuadTreeForest::reserve_nonroot_nodes(int32 new_cap)
  /// {
  /// }


  // execute_refinements()
  void QuadTreeForest::execute_refinements()
  {
    throw std::logic_error("Not implemented: execute_refinements()");

    this->m_leafs.reset();
  }

  // IndexLeafs recipe
  Array<int32> QuadTreeForest::IndexLeafs::operator()
    (const QuadTreeForest *forest) const
  {
    DeviceQuadTreeForest d_forest(*forest);
    return array_where_true(forest->num_nodes(),
        [=] DRAY_LAMBDA (int32 ii) { return d_forest.leaf(ii); });
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
    ConstDeviceArray<int32> d_leafs(this->m_leafs.get());
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
      Vec<Vec<Float, ncomp>, 3> vol_jacobian = phi_prime(q.center().m_loc);
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
        m_first_child(host.m_first_child),
        m_parent(host.m_parent),
        m_valid(host.m_valid)
    { }


}//namespace dray

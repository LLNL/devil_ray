// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/types.hpp>
#include <dray/array.hpp>
#include <dray/array_utils.hpp>

namespace dray
{
  namespace detail
  {
    // trapezoid()
    template <class DeviceLocationToJacobian,
             class DeviceFaceLocationToScalar>
    DRAY_EXEC Float trapezoid(
        const Quadrant &q,
        const DeviceLocationToJacobian &phi_prime,
        const DeviceFaceLocationToScalar &integrand);

    // estimate_error()
    DRAY_EXEC ValueError estimate_error(
        TreeNodePtr node,
        TreeNodePtr parent,
        TreeNodePtr grandparent,
        const ConstDeviceArray<Float> &d_node_value,
        const ConstDeviceArray<Float> &d_node_sum_of_children);
  }


  // pagani_iteration()  (templated factory method)
  template <class DeviceLocationToJacobian,
           class DeviceFaceLocationToScalar>
  PaganiIteration<DeviceLocationToJacobian,
                  DeviceFaceLocationToScalar>
  pagani_iteration(
      Array<FaceLocation> face_centers,
      const DeviceLocationToJacobian &phi_prime,
      const DeviceFaceLocationToScalar &integrand,
      const Float rel_err_tol,
      const int32 nodes_max,
      const int32 iter_max)
  {
    return PaganiIteration<DeviceLocationToJacobian,
                           DeviceFaceLocationToScalar>
        (face_centers, phi_prime, integrand, rel_err_tol, nodes_max, iter_max);
  }


  // pagani_phys_area_to_mesh()
  template <class DeviceLocationToJacobian,
           class DeviceFaceLocationToScalar>
  IntegrateToMesh pagani_phys_area_to_mesh(
      Array<FaceLocation> face_centers,
      const DeviceLocationToJacobian &phi_prime,
      const DeviceFaceLocationToScalar &integrand,
      const Float rel_err_tol,
      const int32 nodes_max,
      const int32 iter_max)
  {
    PaganiIteration<DeviceLocationToJacobian, DeviceFaceLocationToScalar>
        pagani(
            face_centers, phi_prime, integrand, rel_err_tol, nodes_max, iter_max);

    while (pagani.need_more_refinements())
      pagani.execute_refinements();

    IntegrateToMesh integrate_to_mesh;
    integrate_to_mesh.m_result = pagani.value_error().value();
    return integrate_to_mesh;
  }

  // PaganiIteration() (constructor)
  template <class DL2J, class DFL2S>
  PaganiIteration<DL2J, DFL2S>::PaganiIteration(
      Array<FaceLocation> face_centers,
      const DL2J &phi_prime,
      const DFL2S &integrand,
      const Float rel_err_tol,
      const int32 nodes_max,
      const int32 iter_max)
    :
      m_face_centers(face_centers),
      m_phi_prime(phi_prime),
      m_integrand(integrand),
      m_rel_err_tol(rel_err_tol),
      m_nodes_max(nodes_max),
      m_iter_max(iter_max)
  {
    const int32 num_faces = face_centers.size();
    if (num_faces > nodes_max)
    {
      DRAY_ERROR("pagani_phys_area_to_mesh: node budget smaller than mesh faces");
    }

    m_forest.resize(num_faces);

    // Persistent nodal arrays, expanded across iterations
    m_node_value.resize(m_forest.num_nodes());
    m_node_sum_of_children.resize(m_forest.num_nodes());
    m_node_error.resize(m_forest.num_nodes());
    array_memset_zero(m_node_sum_of_children);
    array_memset(m_node_error, nan<Float>());

    // new_node_list: "May need to refine"
    m_new_node_list = array_counting(m_forest.num_nodes(), 0, 1);

    fprintf(stderr, "rel_tol=%.1e  nodes_max=%.0e  iter_max=%d\n",
        rel_err_tol, Float(nodes_max), iter_max);
    fprintf(stderr, "%7s %8s %8s %8s %8s %12s %12s\n",
        "iters", "nodes", "leafs", "over", "new", "value", "error");

    m_partial = {0, 0};  // accumulation of finalized regions.
    m_iter = 0;
    m_stage = UninitLeafs;
    m_working = {nan<Float>(), nan<Float>()};
    m_finalizing = {nan<Float>(), nan<Float>()};
  }

  // need_more_refinements()
  template <class DL2J, class DFL2S>
  bool PaganiIteration<DL2J, DFL2S>::need_more_refinements() const
  {
    // Calls eval_values() and eval_error_and_refinements().
    ready_refinements();

    const int32 refined_size =
        m_forest.num_nodes() + m_count_refinements * QuadTreeForest::NUM_CHILDREN;

    return (m_iter < m_iter_max &&
            refined_size <= m_nodes_max &&
            m_count_refinements > 0);
  }

  // eval_values()
  template <class DL2J, class DFL2S>
  void PaganiIteration<DL2J, DFL2S>::eval_values() const
  {
    ConstDeviceArray<FaceLocation> d_face_centers(m_face_centers);
    ConstDeviceArray<TreeNodePtr> d_new_node_list(m_new_node_list);
    DeviceQuadTreeForest d_forest(m_forest);

    const DL2J &phi_prime = m_phi_prime;
    const DFL2S &integrand = m_integrand;

    NonConstDeviceArray<Float> d_node_value(m_node_value);
    NonConstDeviceArray<Float> d_node_sum_of_children(m_node_sum_of_children);

    RAJA::ReduceSum<reduce_policy, IntegrateT> sum_new_values(0.0f);
    RAJA::ReduceSum<reduce_policy, IntegrateT> sum_parent_values(0.0f);

    // First pass:
    //   - For all new leafs
    //     - Evaluate sub-integral
    //     - Update parent sum-of-children
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, m_new_node_list.size()),
        [=] DRAY_LAMBDA (int32 i)
    {
      const TreeNodePtr new_leaf = d_new_node_list.get_item(i);
      const QuadTreeQuadrant qt_quadrant = d_forest.quadrant(new_leaf);
      const FaceLocation face_center =
          d_face_centers.get_item(qt_quadrant.tree_id());
      const Quadrant quadrant = Quadrant::create(face_center, qt_quadrant);

      // evaluate leaf
      const Float q_value =
          detail::trapezoid(quadrant, phi_prime, integrand);
      d_node_value.get_item(new_leaf) = q_value;

      sum_new_values += q_value;

      // update parent
      if (!d_forest.root(new_leaf))
      {
        const TreeNodePtr parent = d_forest.parent(new_leaf);
        RAJA::atomicAdd<atomic_policy>(
            &d_node_sum_of_children.get_item(parent), q_value);

        if (d_forest.child_num(new_leaf) == 0)
          sum_parent_values += d_node_value.get_item(parent);
      }
    });

    m_working.m_value = sum_new_values.get();

    m_stage = EvaldVals;
  }


  // eval_error_and_refinements()
  template <class DL2J, class DFL2S>
  void PaganiIteration<DL2J, DFL2S>::eval_error_and_refinements() const
  {
    const int32 num_nodes = m_forest.num_nodes();
    const int32 num_leafs = m_forest.num_leafs();
    const int32 num_new = m_new_node_list.size();

    if (m_iter <= 1)  // No grandparents yet, so refine everywhere.
    {
      m_refinements.resize(num_nodes);
      array_memset(m_refinements, 1);
      m_count_refinements = num_nodes;

      m_stage = EvaldError;

      m_finalizing.m_value = 0;
      m_finalizing.m_error = 0;
    }
    else  // Deep enough to have grandparents, to calculate error.
    {
      const Float rel_tol = m_rel_err_tol;
      /// ConstDeviceArray<TreeNodePtr> d_leafs(m_forest.leafs());
      ConstDeviceArray<TreeNodePtr> d_new_node_list(m_new_node_list);
      DeviceQuadTreeForest d_forest(m_forest);
      ConstDeviceArray<Float> d_node_value(m_node_value);
      ConstDeviceArray<Float> d_node_sum_of_children(m_node_sum_of_children);
      NonConstDeviceArray<Float> d_node_error(m_node_error);

      RAJA::ReduceSum<reduce_policy, IntegrateT> sum_new_errs(0.0f);
      RAJA::ReduceMin<reduce_policy, Float> min_error(infinity<Float>());
      RAJA::ReduceMax<reduce_policy, Float> max_error(-infinity<Float>());

      m_refinements.resize(num_nodes);
      array_memset_zero(m_refinements);
      NonConstDeviceArray<int32> d_refinements(m_refinements);
      RAJA::ReduceSum<reduce_policy, int32> count_refine(0);

      RAJA::ReduceSum<reduce_policy, IntegrateT> sum_finalized_values(0.0f);
      RAJA::ReduceSum<reduce_policy, IntegrateT> sum_finalized_errors(0.0f);

      // Second pass (relative-error classify):
      // - For all new leafs
      //   - Evaluate error heuristic
      //   - Reduce-sum and mark refines of (region rel err > rel tolerance)
      RAJA::forall<for_policy>(RAJA::RangeSegment(0, m_new_node_list.size()),
          [=] DRAY_LAMBDA (int32 i)
      {
        const TreeNodePtr new_leaf = d_new_node_list.get_item(i);
        const TreeNodePtr parent = d_forest.parent(new_leaf);
        const TreeNodePtr grandparent = d_forest.parent(parent);

        // Evaluate error.
        const ValueError leaf_error = detail::estimate_error(
            new_leaf, parent, grandparent, d_node_value, d_node_sum_of_children);
        d_node_error.get_item(new_leaf) = leaf_error.absolute();
        sum_new_errs += leaf_error.absolute();

        min_error.min(leaf_error.absolute());
        max_error.max(leaf_error.absolute());

        // Classification: Compare relative error.
        if (leaf_error.relative() > rel_tol)
        {
          d_refinements.get_item(new_leaf) = true;
          count_refine += 1;
        }
        else
        {
          sum_finalized_values += leaf_error.value();
          sum_finalized_errors += leaf_error.absolute();
        }
      });

      m_working.m_error = sum_new_errs.get();
      m_count_refinements = count_refine.get();

      m_stage = EvaldError;  // can value_error() without infinite recursion.

      const ValueError total = this->value_error();

      /// // Third pass(es) to amend error or memory (threshold error classify):
      /// if (relative(delta(total.value())) < tau_rel && total.relative() > tau_rel
      ///     || m_count_refinements > able_to_refine)
      /// {
      //
      /// sum_finalized_values.reset(0.0f);
      /// sum_finalized_errors.reset(0.0f);
      ///   throw std::logic_error("not implemented");
      /// }

      m_finalizing.m_value = sum_finalized_values.get();
      m_finalizing.m_error = sum_finalized_errors.get();
    }

    m_stage = EvaldRefines;

    const ValueError total = this->value_error();
    fprintf(stderr, "%7d %8d %8d %7.1f%% %8d %12f %12e\n",
        m_iter+1,
        num_nodes,
        num_leafs,
        100.*(num_leafs - num_new)/num_leafs,
        m_count_refinements * QuadTreeForest::NUM_CHILDREN,
        total.m_value,
        total.m_error);
  }

  // execute_refinements()
  template <class DL2J, class DFL2S>
  void PaganiIteration<DL2J, DFL2S>::execute_refinements()
  {
    ready_refinements();

    const int32 num_nodes = m_forest.num_nodes();

    QuadTreeForest::ExpansionPlan plan = m_forest.execute_refinements(m_refinements);
    m_forest.expand_node_array(plan, m_node_value, Float(0));
    m_forest.expand_node_array(plan, m_node_sum_of_children, Float(0));
    m_forest.expand_node_array(plan, m_node_error, nan<Float>());

    m_new_node_list = array_counting(
        m_forest.num_nodes() - num_nodes, num_nodes, 1);

    m_partial.m_value += m_finalizing.value();
    m_partial.m_error += m_finalizing.absolute();

    m_iter++;
    m_stage = UninitLeafs;
    m_working = {nan<Float>(), nan<Float>()};
    m_finalizing = {nan<Float>(), nan<Float>()};
  }


  // leaf_values()
  template <class DL2J, class DFL2S>
  Array<Float> PaganiIteration<DL2J, DFL2S>::leaf_values() const
  {
    ready_values();
    Array<Float> leaf_values = gather(m_node_value, m_forest.leafs());
    return leaf_values;
  }

  // leaf_error()
  template <class DL2J, class DFL2S>
  Array<Float> PaganiIteration<DL2J, DFL2S>::leaf_error() const
  {
    ready_error();
    Array<Float> leaf_error = gather(m_node_error, m_forest.leafs());
    return leaf_error;
  }

  // value_error()
  template <class DL2J, class DFL2S>
  ValueError PaganiIteration<DL2J, DFL2S>::value_error() const
  {
    ready_error();
    ValueError total =
        { m_partial.value() + m_working.value(),
          m_partial.absolute() + m_working.absolute() };
    return total;
  }

  // forest()
  template <class DL2J, class DFL2S>
  const QuadTreeForest & PaganiIteration<DL2J, DFL2S>::forest() const
  {
    return m_forest;
  }

  // ready_values()
  template <class DL2J, class DFL2S>
  void PaganiIteration<DL2J, DFL2S>::ready_values() const
  {
    if (m_stage < EvaldVals)
      eval_values();
  }

  // ready_error()
  template <class DL2J, class DFL2S>
  void PaganiIteration<DL2J, DFL2S>::ready_error() const
  {
    ready_values();
    if (m_stage < EvaldError)
      eval_error_and_refinements();
    // current implementation forces computing both error and refinements.
  }

  // ready_refinements()
  template <class DL2J, class DFL2S>
  void PaganiIteration<DL2J, DFL2S>::ready_refinements() const
  {
    ready_error();
    if (m_stage < EvaldRefines)
      eval_error_and_refinements();
  }


  namespace detail
  {
    // trapezoid()
    template <class DeviceLocationToJacobian,
             class DeviceFaceLocationToScalar>
    DRAY_EXEC Float trapezoid(
        const Quadrant &q,
        const DeviceLocationToJacobian &phi_prime,
        const DeviceFaceLocationToScalar &integrand)
    {
      // Evaluate physical-space area of the (sub)quad.
      const Float dA = q.world_area(phi_prime(q.center().loc()));

      // Integrate (trapezoidal rule).
      Float region_value = 0.0f;
      region_value += integrand(q.lower_left());
      region_value += integrand(q.lower_right());
      region_value += integrand(q.upper_left());
      region_value += integrand(q.upper_right());
      region_value /= 4;
      const Float region_integral = region_value * dA;

      return region_integral;
    }


    // Custom error estimate for a quadrant of side length (h):
    //   Assume val(h) is the integral value computed at resolution h.
    //   Assume val(*) is the true integral value.
    //   Define err(h) := val(h) - val(*)
    //
    //   Use the ansatz  err(h) = K * err(2h)  for some constant K.
    //   Compute dlt(h,2h) := val(h) - val(2h)
    //                     == err(h) - err(2h)
    //   With three resolutions, {(h) (2h) (4h)}, solve for err(h) and K.
    //     K      == dlt(h,2h) / dlt(2h,4h)
    //     err(h) == dlt(h,2h)^2 / (dlt(h,2h) - dlt(2h,4h))

    // estimate_error()
    DRAY_EXEC ValueError estimate_error(
        TreeNodePtr node,
        TreeNodePtr parent,
        TreeNodePtr grandparent,
        const ConstDeviceArray<Float> &d_node_value,
        const ConstDeviceArray<Float> &d_node_sum_of_children)
    {
      const Float int2 = d_node_value.get_item(grandparent);
      const Float int1 = d_node_value.get_item(parent);
      const Float int0 = d_node_value.get_item(node);
      const Float sum0 = d_node_sum_of_children.get_item(parent);
      const Float sum1 = d_node_sum_of_children.get_item(grandparent);

      // node integral.
      const Float v00 = int0;
      // fraction of parent integral.
      const Float v01 = int1 * (int0 * rcp_safe(sum0));
      // fraction of grandparent integral.
      const Float v02 = int2 * (int1 * rcp_safe(sum1)) * (int0 * rcp_safe(sum0));

      ValueError value_error;
      value_error.m_value = v00;
      if (v00 - v01 == v01 - v02)
        value_error.m_error = fabs(v00 - v01);
      else
        value_error.m_error = fabs((v00 - v01) * (v00 - v01) / (v00 - 2 * v01 + v02));

      return value_error;
    }
  }

}

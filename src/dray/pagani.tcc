// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

//#include <dray/pagani.hpp>
#include <dray/types.hpp>
#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/utils/data_logger.hpp>

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
        const NonConstDeviceArray<Float> &d_node_value,
        const NonConstDeviceArray<Float> &d_node_sum_of_children);
  }


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
    const int32 num_faces = face_centers.size();
    if (num_faces > nodes_max)
    {
      DRAY_ERROR("pagani_phys_area_to_mesh: node budget smaller than mesh faces");
    }

    // Initialize quadtree forest and nodal arrays.
    QuadTreeForest forest;
    forest.resize(num_faces);

    // Persistent nodal arrays, expanded across iterations
    Array<Float> node_value;
    Array<Float> node_sum_of_children;  // for custom error estimate.
    node_value.resize(forest.num_nodes());
    node_sum_of_children.resize(forest.num_nodes());
    array_memset_zero(node_sum_of_children);

    // Overwritten each iteration
    Array<int32> node_refine;
    Array<Float> node_error;

    DRAY_LOG_OPEN("pagani");

    // new_node_list: "May need to refine"
    Array<int32> new_node_list = array_counting(forest.num_nodes(), 0, 1);

    fprintf(stderr, "rel_tol=%.1e  nodes_max=%.0e  iter_max=%d\n",
        rel_err_tol, Float(nodes_max), iter_max);
    fprintf(stderr, "%7s %8s %8s %8s %8s %12s %12s\n",
        "iters", "nodes", "leafs", "over", "new", "value", "error");

    // Refinement loop.
    ValueError total = {0, 0};  // iterative integral estimate.
    const Float rel_tol = rel_err_tol;
    int32 iter = 0;
    while (iter < iter_max &&
           new_node_list.size() > 0)
    {
      const int32 num_nodes = forest.num_nodes();
      const int32 num_leafs = forest.num_leafs();
      const int32 num_new = new_node_list.size();
      ConstDeviceArray<FaceLocation> d_face_centers(face_centers);
      ConstDeviceArray<TreeNodePtr> d_leafs(forest.leafs());
      ConstDeviceArray<TreeNodePtr> d_new_node_list(new_node_list);
      DeviceQuadTreeForest d_forest(forest);

      NonConstDeviceArray<Float> d_node_value(node_value);
      NonConstDeviceArray<Float> d_node_sum_of_children(node_sum_of_children);

      RAJA::ReduceSum<reduce_policy, IntegrateT> sum_new_values(0.0f);
      RAJA::ReduceSum<reduce_policy, IntegrateT> sum_parent_values(0.0f);

      // First pass:
      //   - For all new leafs
      //     - Evaluate sub-integral
      //     - Update parent sum-of-children
      RAJA::forall<for_policy>(RAJA::RangeSegment(0, new_node_list.size()), [=] DRAY_LAMBDA (int32 i)
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

      RAJA::ReduceSum<reduce_policy, IntegrateT> sum_errors(0.0f);

      if (iter <= 1)  // No grandparents yet, so refine everywhere.
      {
        Array<int32> refinements;
        refinements.resize(num_nodes);
        array_memset(refinements, 1);

        QuadTreeForest::ExpansionPlan plan = forest.execute_refinements(refinements);
        forest.expand_node_array(plan, node_value, Float(0));
        forest.expand_node_array(plan, node_sum_of_children, Float(0));
      }
      else  // Deep enough to have grandparents, to calculate error.
      {
        RAJA::ReduceSum<reduce_policy, int32> count_refine(0);
        Array<int32> refinements;
        refinements.resize(num_nodes);
        array_memset_zero(refinements);
        NonConstDeviceArray<int32> d_refinements(refinements);

        // Second pass (relative-error classify):
        // - For all leafs
        //   - Evaluate error heuristic
        //   - Reduce-sum and prefix-sum (region rel err > rel tolerance)
        //   - Reduce-sums (region abs error > abs thresholds)
        RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_leafs), [=] DRAY_LAMBDA (int32 i)
        {
          const TreeNodePtr leaf = d_leafs.get_item(i);
          const TreeNodePtr parent = d_forest.parent(leaf);
          const TreeNodePtr grandparent = d_forest.parent(parent);

          const ValueError leaf_error = detail::estimate_error(
              leaf, parent, grandparent, d_node_value, d_node_sum_of_children);

          const Float err_abs = leaf_error.absolute();
          const Float err_rel = leaf_error.relative();

          // Compare relative error.
          if (err_rel > rel_tol)
          {
            d_refinements.get_item(leaf) = true;
            count_refine += 1;
          }

          sum_errors += err_abs;

          //TODO abs threshold
        });

        // Third pass(es) (threshold error classify):

        // Expand and refine.
        if (num_nodes + count_refine.get() <= nodes_max)
        {
          QuadTreeForest::ExpansionPlan plan = forest.execute_refinements(refinements);
          forest.expand_node_array(plan, node_value, Float(0));
          forest.expand_node_array(plan, node_sum_of_children, Float(0));
        }
      }

      total.m_value += sum_new_values.get() - sum_parent_values.get();
      total.m_error = sum_errors.get(); 

      // Update new_node_list
      new_node_list = array_counting(forest.num_nodes() - num_nodes, num_nodes, 1);
      ++iter;

      fprintf(stderr, "%7d %8d %8d %7.1f%% %8d %12f %12e\n",
          iter, num_nodes, num_leafs, 100.*(num_leafs - num_new)/num_leafs, int32(new_node_list.size()), total.m_value, total.m_error);

      if (iter > 2)
      {
        DRAY_LOG_ENTRY("iterations", iter);
        DRAY_LOG_ENTRY("num_nodes", num_nodes);
        DRAY_LOG_ENTRY("error", total.m_error);
      }
    }

    DRAY_LOG_CLOSE();

    IntegrateToMesh integration_result;
    integration_result.m_result = total.m_value;
    return integration_result;
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
        const NonConstDeviceArray<Float> &d_node_value,
        const NonConstDeviceArray<Float> &d_node_sum_of_children)
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

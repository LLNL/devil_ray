// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_PAGANI_HPP
#define DRAY_PAGANI_HPP

/**
 * Algorithm is based on the following paper:
 *   PAGANI: A Parallel Adaptive GPU Algorithm for Numerical Integration
 *   Sakiotis, Arumugam, Paterno, Ranjjan, Terzic, Zubair. 2021.
 *   (https://arxiv.org/abs/2104.06494)
 *
 * @author Masado Ishii
 * @note My implementation differs from the paper as follows:
 *       - I use quadtrees; they use k-d trees.
 *       - I retain internal nodes and finalized leafs;
 *         they retain only active leafs.
 *       - I use a simplistic error estimate;
 *         they use a combination of error estimates.
 */

#include <dray/quadtree.hpp>
#include <dray/types.hpp>
#include <dray/integrate.hpp>

namespace dray
{
  struct ValueError
  {
    Float m_value;
    Float m_error;
    Float value() const { return m_value; }
    Float absolute() const { return m_error; }
    Float relative() const { return m_error / m_value; }
  };

  // TODO return error too


  // pagani_phys_area_to_mesh()
  template <class DeviceLocationToJacobian,
           class DeviceFaceLocationToScalar>
  IntegrateToMesh pagani_phys_area_to_mesh(
      Array<FaceLocation> face_centers,
      const DeviceLocationToJacobian &phi_prime,
      const DeviceFaceLocationToScalar &integrand,
      const Float rel_err_tol,
      const int32 nodes_max,
      const int32 iter_max);


  // PaganiIteration
  template <class DeviceLocationToJacobian,
           class DeviceFaceLocationToScalar>
  class PaganiIteration
  {
    public:
      PaganiIteration( Array<FaceLocation> face_centers,
                       const DeviceLocationToJacobian &phi_prime,
                       const DeviceFaceLocationToScalar &integrand,
                       const Float rel_err_tol,
                       const int32 nodes_max,
                       const int32 iter_max);

      bool need_more_refinements() const;  //TODO decouple need more refines from iter_max and nodes_max, make it relative to current rel_err_tol
      void execute_refinements();
      Array<Float> leaf_values() const;
      Array<Float> leaf_error() const;
      ValueError value_error() const;
      const QuadTreeForest & forest() const;

    protected:
      void ready_values() const;
      void ready_error() const;
      void ready_refinements() const;

      const Array<FaceLocation> m_face_centers;
      const DeviceLocationToJacobian & m_phi_prime;
      const DeviceFaceLocationToScalar & m_integrand;
      const Float m_rel_err_tol;
      const int32 m_nodes_max;
      const int32 m_iter_max;

      enum Stage { UninitLeafs = 0, EvaldVals, EvaldError, EvaldRefines };
      // UninitLeafs:  New leafs are uninitd in m_node_value,
      //               parents unintd in m_node_sum_of_children.
      // EvaldVals:    New leafs and parents initialized,
      //               but error not computed and refinements not computed.
      // EvaldError:   Error computed, updated m_total,
      //               but refinements not computed (not reachable).
      // EvaldRefines: Updated m_total and refinements ready to execute.

      int32 m_iter;
      mutable Stage m_stage;
      mutable ValueError m_total;
      mutable QuadTreeForest m_forest;
      mutable Array<Float> m_node_value;
      mutable Array<Float> m_node_sum_of_children;  // for custom error estimate
      mutable Array<int32> m_new_node_list;
      mutable Array<int32> m_refinements;
      mutable int32 m_count_refinements;

    private:
      void eval_values() const;
      void eval_error_and_refinements() const;
  };



}//namespace dray

#include <dray/pagani.tcc>

#endif//DRAY_PAGANI_HPP


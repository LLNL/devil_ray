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

  template <class DeviceLocationToJacobian,
           class DeviceFaceLocationToScalar>
  IntegrateToMesh pagani_phys_area_to_mesh(
      Array<FaceLocation> face_centers,
      const DeviceLocationToJacobian &phi_prime,
      const DeviceFaceLocationToScalar &integrand,
      const Float rel_err_tol,
      const int32 nodes_max,
      const int32 iter_max);

}//namespace dray

#include <dray/pagani.tcc>

#endif//DRAY_PAGANI_HPP


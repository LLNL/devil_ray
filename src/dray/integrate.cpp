// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/integrate.hpp>
#include <dray/array_utils.hpp>

namespace dray
{
  /// Array<FaceLocation> Integrator::locations() const
  /// {
  ///   throw std::logic_error("Not implemented: locations()");
  /// }

  /// Array<Float> Integrator::weights() const
  /// {
  ///   throw std::logic_error("Not implemented: weights()");
  /// }


  Float IntegrateToMesh::result() const
  {
    return m_result;
  }

  Array<Float> IntegrateToFaces::result() const
  {
    return m_result;
  }

  /// void IntegrateToMesh::ref_area(
  ///     Mesh &mesh, const Integrator &integrator)
  /// {
  ///   Array<Float> weights = integrator.weights();
  ///   m_result = array_sum(weights);
  /// }

  /// void IntegrateToMesh::phys_area(
  ///     Mesh &mesh, const Integrator &integrator)
  /// {
  ///   Array<FaceLocation> locations = integrator.locations();
  ///   Array<Float> weights = integrator.weights();

  ///   throw std::logic_error("Not implemented: phys_area()");
  /// }

  /// void IntegrateToFaces::ref_area(
  ///     Mesh &mesh, const Integrator &integrator)
  /// {
  ///   Array<Float> weights = integrator.weights();
  ///   // TODO integrate from quadtree quads to faces, should add up to 1.0 in each

  ///   throw std::logic_error("Not implemented: ref_area()");
  /// }

  /// void IntegrateToFaces::phys_area(
  ///     Mesh &mesh, const Integrator &integrator)
  /// {
  ///   Array<FaceLocation> locations = integrator.locations();
  ///   Array<Float> weights = integrator.weights();

  ///   throw std::logic_error("Not implemented: phys_area()");
  /// }


}//namespace dray

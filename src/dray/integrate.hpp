// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_INTEGRATE_HPP
#define DRAY_INTEGRATE_HPP

#include <dray/types.hpp>
#include <dray/array.hpp>

namespace dray
{
  /// class Integrator
  /// {
  ///   public:
  ///     Array<FaceLocation> locations() const;
  ///     Array<Float> weights() const;

  ///   private:
  ///     QuadTreeForest m_forest;
  /// };

  typedef double IntegrateT;

  // Integrate over whole mesh, getting a single value.
  struct IntegrateToMesh
  {
    /// void ref_area(Mesh &mesh, const Integrator &integrator);
    /// void phys_area(Mesh &mesh, const Integrator &integrator);

    Float m_result;
    Float result() const;
  };

  // Integrate over each face separately, getting many values.
  struct IntegrateToFaces
  {
    /// void ref_area(Mesh &mesh, const Integrator &integrator);
    /// void phys_area(Mesh &mesh, const Integrator &integrator);

    Array<Float> m_result;
    Array<Float> result() const;
  };

}//namespace dray

#endif//DRAY_INTEGRATE_HPP

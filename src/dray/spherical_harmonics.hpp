// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SPHERICAL_HARMONICS_HPP
#define DRAY_SPHERICAL_HARMONICS_HPP

#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <typename T>
class SphericalHarmonics
{
  public:
    DRAY_EXEC SphericalHarmonics(int32 legendre_order) : m_legendre_order(legendre_order) {}
    DRAY_EXEC ~SphericalHarmonics()
    {
      delete_buffer();
    }

    /** Evaluates all spherical harmonics up to legendre_order. */
    DRAY_EXEC
    const T* eval_all(const Vec<T, 3> &xyz_normal);

    /** Calls eval_all() and performs dot product. */
    DRAY_EXEC
    T eval_function(const T * coefficients, const Vec<T, 3> &xyz_normal)
    {
      return eval_function(m_legendre_order,
                              coefficients,
                              eval_all(xyz_normal));
    }

    /** Calls eval_all() and accumulates vector to coefficients. */
    DRAY_EXEC
    void project_point(T * coefficients,
                       const Vec<T, 3> &xyz_normal,
                       const T integration_value,
                       const T integration_weight)
    {
      project_point(m_legendre_order,
                       coefficients,
                       eval_all(xyz_normal),
                       integration_value,
                       integration_weight);
    }

    DRAY_EXEC int32 num_harmonics() const { return num_harmonics(m_legendre_order); }


    DRAY_EXEC static int32 index(int32 n, int32 m) { return n * (n+1) + m; }
    DRAY_EXEC static int32 alp_index(int32 n, int32 m) { return n * (n+1) / 2 + m; }
    // alp = associated legendre polynomial, only uses m >= 0.

    DRAY_EXEC static int32 num_harmonics(int32 legendre_order)
    {
      return (legendre_order+1)*(legendre_order+1);
    }

    /** Static version does not call eval_all().
     *  Good for evaluating different functions
     *  with different sets of coefficients. */
    DRAY_EXEC
    static T eval_function(const int32 legendre_order,
                         const T * coefficients,
                         const T * sph_harmonics)
    {
      T value = 0.0f;
      const int32 Np1_sq = num_harmonics(legendre_order);
      for (int32 nm = 0; nm < Np1_sq; ++nm)
        value += coefficients[nm] * sph_harmonics[nm];
      return value;
    }

    /** Static version does not call eval_all().
     *  Good for projecting different integration values
     *  to different sets of coefficients. */
    DRAY_EXEC
    static void project_point(const int32 legendre_order,
                              T * coefficients,
                              const T * sph_harmonics,
                              const T integration_value,
                              const T integration_weight)
    {
      const int32 Np1_sq = num_harmonics(legendre_order);
      const T integration_product = integration_value * integration_weight;
      for (int32 nm = 0; nm < Np1_sq; ++nm)
        coefficients[nm] += sph_harmonics[nm] * integration_product;
    }

  private:
    DRAY_EXEC T * resize_buffer(const size_t size);
    DRAY_EXEC void delete_buffer() { if (m_buffer != nullptr) delete [] m_buffer; }

  private:
    int32 m_legendre_order = 0;
    size_t m_buffer_size = 0;
    char * m_buffer = nullptr;
};



}

#endif//DRAY_SPHERICAL_HARMONICS_HPP

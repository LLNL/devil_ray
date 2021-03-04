// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SPHERICAL_HARMONICS_HPP
#define DRAY_SPHERICAL_HARMONICS_HPP

#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

#include <cmath>

namespace dray
{

template <typename T>
class SphericalHarmonics
{
  public:
    DRAY_EXEC SphericalHarmonics(int32 legendre_order) : m_legendre_order(legendre_order) {}
    DRAY_EXEC ~SphericalHarmonics();

    /** Evaluates all spherical harmonics (shape funcs) up to legendre_order. */
    DRAY_EXEC
    const T* eval_all(const Vec<T, 3> &xyz_normal);

    /** Calls eval_all() and performs dot product. */
    DRAY_EXEC
    T eval_function(const T * coefficients, const Vec<T, 3> &xyz_normal);

    /** Calls eval_all() and accumulates vector to coefficients. */
    DRAY_EXEC
    void project_point(T * coefficients,
                       const Vec<T, 3> &xyz_normal,
                       const T integration_value,
                       const T integration_weight);

    DRAY_EXEC int32 num_harmonics() const;


    /** Linearizes index with {0 <= n <= legendre_order}, {-n <= m <= +n}. */
    DRAY_EXEC static int32 index(int32 n, int32 m);

    /** Linearizes index with {0 <= n <= legendre_order}, {0 <= m <= +n}. */
    DRAY_EXEC static int32 alp_index(int32 n, int32 m);
    // alp = associated legendre polynomial, only uses m >= 0.

    DRAY_EXEC static int32 num_harmonics(int32 legendre_order);

    /** Static version does not call eval_all().
     *  Good for evaluating different functions
     *  with different sets of coefficients. */
    DRAY_EXEC
    static T eval_function(const int32 legendre_order,
                         const T * coefficients,
                         const T * sph_harmonics);

    /** Static version does not call eval_all().
     *  Good for projecting different integration values
     *  to different sets of coefficients. */
    DRAY_EXEC
    static void project_point(const int32 legendre_order,
                              T * coefficients,
                              const T * sph_harmonics,
                              const T integration_value,
                              const T integration_weight);

  private:
    DRAY_EXEC T * resize_buffer(const size_t size);
    DRAY_EXEC void delete_buffer();

  private:
    int32 m_legendre_order = 0;
    size_t m_buffer_size = 0;
    char * m_buffer = nullptr;
};



// --------------------------------------------------------------------------
// Implementations
// --------------------------------------------------------------------------

//
// resize_buffer()
//
template <typename T>
DRAY_EXEC
T * SphericalHarmonics<T>::resize_buffer(const size_t size)
{
  size_t new_size = sizeof(T) * size;
  if (m_buffer_size < new_size)
  {
    if (m_buffer != nullptr)
      delete [] m_buffer;
    m_buffer = new char[new_size];
  }
  return (T*)(m_buffer);
}



inline double fact(int a) { return std::tgamma(a+1); }

inline double Knm(int n, int absm)
{
  return std::sqrtl( (2*n+1) * (fact(n-absm) / fact(n+absm)) / (4*pi()) );
}



//
// eval_all()
//
template <typename T>
DRAY_EXEC
const T* SphericalHarmonics<T>::eval_all(const Vec<T, 3> &xyz_normal)
{
  // Computed using the recursive formulation in Appendix A1 in
  //
  //     @inproceedings{sloan2008stupid,
  //       title={Stupid spherical harmonics (sh) tricks},
  //       author={Sloan, Peter-Pike},
  //       booktitle={Game developers conference},
  //       volume={9},
  //       pages={42},
  //       year={2008}
  //     }

  // Note: I came up with a recursive form of the normalization constants K_n^m.
  //   The formula for K_n^m involves ratios of factorials. I used floats
  //   because the ratios do not simply to integers. I haven't studied the stability
  //   properties of evaluating them directly or recursively, so no guarantees.
  //   Also, to test the normalization constants you need to do a reconstruction,
  //   not just evaluate each spherical harmonic individually.

  const int32 Np1 = m_legendre_order + 1;
  const int32 Np1_sq = Np1 * Np1;
  const int32 result_sz = Np1_sq;            // result
  const int32 sin_sz = Np1;                  // sine
  const int32 cos_sz = Np1;                  // cosine
  const int32 alp_sz = Np1 * (Np1+1) / 2;    // associated legendre polynomial
  const int32 k2_sz = Np1 * (Np1+1) / 2;     // square of normalization constant

  T * const buffer = resize_buffer(result_sz + sin_sz + cos_sz + alp_sz + k2_sz);

  T * const resultp = buffer;
  T * const sinp = resultp + result_sz;
  T * const cosp = sinp + sin_sz;
  T * const alpp = cosp + cos_sz;
  T * const k2p = alpp + alp_sz;

  const T sqrt2 = sqrtl(2);

  const T &x = xyz_normal[0];
  const T &y = xyz_normal[1];
  const T &z = xyz_normal[2];

  // m=0
  {
    const int32 m = 0;

    sinp[m] = 0;
    cosp[m] = 1;

    // n == m
    alpp[alp_index(m, m)] = 1;
    k2p[alp_index(0, 0)] = 1.0 / (4 * pi());
    /// resultp[index(m, m)] = sqrt(k2p[alp_index(m, m)]) * alpp[alp_index(m, m)];
    resultp[index(m, m)] = Knm(m, m) * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(1, 0)] = 2 * (1+1) / (4 * pi());
      /// resultp[index(m+1, m)] = sqrt(k2p[alp_index(m+1, m)]) * alpp[alp_index(m+1, m)];
      resultp[index(m+1, m)] = Knm(m+1, m) * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int32 n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, 0)] = (2*n+1) / (4 * pi());

      /// resultp[index(n, m)] = sqrt(k2p[alp_index(n, m)]) * alpp[alp_index(n, m)];
      resultp[index(n, m)] = Knm(n, m) * alpp[alp_index(n, m)];
    }
  }

  // m>0
  for (int32 m = 1; m <= m_legendre_order; ++m)
  {
    sinp[m] = x * sinp[m-1] + y * cosp[m-1];
    cosp[m] = x * cosp[m-1] - y * sinp[m-1];

    // n == m
    alpp[alp_index(m, m)] = (1-2*m) * alpp[alp_index(m-1, m-1)];;
    k2p[alp_index(m, m)] = k2p[alp_index(m-1, m-1)] * (2*m+1) / ((2*m-1) * (2*m-1) * (2*m));
    /// resultp[index(m, m)] = sqrt(2*k2p[alp_index(m, m)]) * cosp[m] * alpp[alp_index(m, m)];
    resultp[index(m, m)] = sqrt2*Knm(m, m) * cosp[m] * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(m+1, m)] =
          k2p[alp_index((m+1)-1, m)] * (2*(m+1)+1) * ((m+1)-m) / ((2*(m+1)-1) * ((m+1)+m));

      /// resultp[index(m+1, m)] = sqrt(2*k2p[alp_index(m+1, m)]) * cosp[m] * alpp[alp_index(m+1, m)];
      resultp[index(m+1, m)] = sqrt2*Knm(m+1, m) * cosp[m] * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int32 n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, m)] = k2p[alp_index(n-1, m)] * 2*(n+1) * (n-m) / ((2*n-1) * (n+m));

      /// resultp[index(n, m)] = sqrt(2*k2p[alp_index(n, m)]) * cosp[m] * alpp[alp_index(n, m)];
      resultp[index(n, m)] = sqrt2*Knm(n, m) * cosp[m] * alpp[alp_index(n, m)];
    }
  }

  // m<0
  for (int32 m = -1; m >= -m_legendre_order; --m)
  {
    const int32 absm = -m;
    for (int32 n = absm; n <= m_legendre_order; ++n)
    {
      /// resultp[index(n, m)] = sqrt(2*k2p[alp_index(n, absm)]) * sinp[absm] * alpp[alp_index(n, absm)];
      resultp[index(n, m)] = sqrt2*Knm(n, absm) * sinp[absm] * alpp[alp_index(n, absm)];
    }
  }

  return resultp;
}


// ~SphericalHarmonics (destructor)
template <typename T>
DRAY_EXEC SphericalHarmonics<T>::~SphericalHarmonics()
{
  delete_buffer();
}


//
// eval_function()
//
template <typename T>
DRAY_EXEC
T SphericalHarmonics<T>::eval_function(const T * coefficients,
                                       const Vec<T, 3> &xyz_normal)
{
  return eval_function(m_legendre_order,
                          coefficients,
                          eval_all(xyz_normal));
}


//
// project_point()
//
template <typename T>
DRAY_EXEC
void SphericalHarmonics<T>::project_point(T * coefficients,
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


//
// num_harmonics()
//
template <typename T>
DRAY_EXEC int32 SphericalHarmonics<T>::num_harmonics() const
{
  return num_harmonics(m_legendre_order);
}


//
// index()  (static)
//
template <typename T>
DRAY_EXEC int32 SphericalHarmonics<T>::index(int32 n, int32 m)
{
  return n * (n+1) + m;
}


//
// alp_index()  (static)
//
template <typename T>
DRAY_EXEC int32 SphericalHarmonics<T>::alp_index(int32 n, int32 m)
{
  return n * (n+1) / 2 + m;
}


//
// num_harmonics()  (static)
//
template <typename T>
DRAY_EXEC int32 SphericalHarmonics<T>::num_harmonics(int32 legendre_order)
{
  return (legendre_order+1)*(legendre_order+1);
}


//
// eval_function()  (static)
//
template <typename T>
DRAY_EXEC
T SphericalHarmonics<T>::eval_function(const int32 legendre_order,
                                       const T * coefficients,
                                       const T * sph_harmonics)
{
  T value = 0.0f;
  const int32 Np1_sq = num_harmonics(legendre_order);
  for (int32 nm = 0; nm < Np1_sq; ++nm)
    value += coefficients[nm] * sph_harmonics[nm];
  return value;
}


//
// project_point (static)
//
template <typename T>
DRAY_EXEC
void SphericalHarmonics<T>::project_point(const int32 legendre_order,
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


//
// delete_buffer()
//
template <typename T>
DRAY_EXEC void SphericalHarmonics<T>::delete_buffer()
{
  if (m_buffer != nullptr) delete [] m_buffer;
}



} // namespace dray

#endif//DRAY_SPHERICAL_HARMONICS_HPP

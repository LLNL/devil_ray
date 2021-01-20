
#include <dray/spherical_harmonics.hpp>

namespace dray
{

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
    resultp[index(m, m)] = sqrt(k2p[alp_index(m, m)]) * alpp[alp_index(m, m)];
    /// resultp[index(m, m)] = Knm(m, m) * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(1, 0)] = 2 * (1+1) / (4 * pi());
      resultp[index(m+1, m)] = sqrt(k2p[alp_index(m+1, m)]) * alpp[alp_index(m+1, m)];
      /// resultp[index(m+1, m)] = Knm(m+1, m) * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int32 n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, 0)] = (2*n+1) / (4 * pi());

      resultp[index(n, m)] = sqrt(k2p[alp_index(n, m)]) * alpp[alp_index(n, m)];
      /// resultp[index(n, m)] = Knm(n, m) * alpp[alp_index(n, m)];
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
    resultp[index(m, m)] = sqrt(2*k2p[alp_index(m, m)]) * cosp[m] * alpp[alp_index(m, m)];
    /// resultp[index(m, m)] = sqrt2*Knm(m, m) * cosp[m] * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(m+1, m)] =
          k2p[alp_index((m+1)-1, m)] * (2*(m+1)+1) * ((m+1)-m) / ((2*(m+1)-1) * ((m+1)+m));

      resultp[index(m+1, m)] = sqrt(2*k2p[alp_index(m+1, m)]) * cosp[m] * alpp[alp_index(m+1, m)];
      /// resultp[index(m+1, m)] = sqrt2*Knm(m+1, m) * cosp[m] * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int32 n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, m)] = k2p[alp_index(n-1, m)] * 2*(n+1) * (n-m) / ((2*n-1) * (n+m));

      resultp[index(n, m)] = sqrt(2*k2p[alp_index(n, m)]) * cosp[m] * alpp[alp_index(n, m)];
      /// resultp[index(n, m)] = sqrt2*Knm(n, m) * cosp[m] * alpp[alp_index(n, m)];
    }
  }

  // m<0
  for (int32 m = -1; m >= -m_legendre_order; --m)
  {
    const int32 absm = -m;
    for (int32 n = absm; n <= m_legendre_order; ++n)
    {
      resultp[index(n, m)] = sqrt(2*k2p[alp_index(n, absm)]) * sinp[absm] * alpp[alp_index(n, absm)];
      /// resultp[index(n, m)] = sqrt2*Knm(n, absm) * sinp[absm] * alpp[alp_index(n, absm)];
    }
  }

  return resultp;
}


// Explicit instantiations
}

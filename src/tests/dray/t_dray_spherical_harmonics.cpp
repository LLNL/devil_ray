// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/device_framebuffer.hpp>

#include <array>
#include <cmath>

class CubeMapConverter
{
  public:
    CubeMapConverter(const int side_length) : m_side_length(side_length) {}
    CubeMapConverter() = default;
    CubeMapConverter(const CubeMapConverter&) = default;
    CubeMapConverter(CubeMapConverter&&) = default;
    CubeMapConverter & operator=(const CubeMapConverter&) = default;
    CubeMapConverter & operator=(CubeMapConverter&&) = default;

    enum Face
    {
      X_plus, X_minus, Y_plus, Y_minus, Z_plus, Z_minus,
      NUM_FACES,
      void_face
    };

    static constexpr std::array<Face, NUM_FACES> get_face_list()
    {
      return {X_plus, X_minus, Y_plus, Y_minus, Z_plus, Z_minus};
    };

    struct UV
    {
      int m_u;
      int m_v;

      UV operator+ (const UV &other) const
      {
        return UV{m_u + other.m_u, m_v + other.m_v};
      }
      UV operator- (const UV &other) const
      {
        return UV{m_u - other.m_u, m_v - other.m_v};
      }
      UV operator* (const int factor) const
      {
        return UV{m_u * factor, m_v * factor};
      }
      UV operator/ (const int factor) const
      {
        return UV{m_u / factor, m_v / factor};
      }
      bool operator==(const UV &other) const
      {
        return (m_u == other.m_u && m_v == other.m_v);
      }
      bool operator!=(const UV &other) const
      {
        return ! (*this == other);
      }
    };

    struct FaceUV
    {
      Face m_face;
      UV m_uv;

      bool is_void () const { return m_face == void_face; }
    };

    UV face_origin(const Face face) const
    {
      return face_unit_origin(face) * m_side_length;
    }

    UV faceuv_to_uv(const FaceUV &faceuv) const
    {
      return face_origin(faceuv.m_face) + faceuv.m_uv;
    }

    FaceUV uv_to_faceuv(const UV &uv) const
    {
      const UV unit_origin = uv / m_side_length;
      const UV face_offset = uv - unit_origin * m_side_length;

      for (Face face : get_face_list())
        if (unit_origin == face_unit_origin(face))
          return FaceUV{face, face_offset};

      return FaceUV{void_face, face_offset};
    }

    UV get_extent() const
    {
      return unit_extent() * m_side_length;
    }

    template <typename T = float>
    dray::Vec<T, 3> to_vec(const FaceUV faceuv) const
    {
      const Face face = faceuv.m_face;
      const int u = faceuv.m_uv.m_u;
      const int v = faceuv.m_uv.m_v;

      const T pm = (face == X_plus || face == Y_plus || face == Z_plus ?
        1.0f : -1.0f);

      const T uf = (2*u - m_side_length) / double(m_side_length);
      const T vf = (2*v - m_side_length) / double(m_side_length);

      dray::Vec<T, 3> xyz;

      switch(face)
      {
        case X_plus:  xyz = {{pm, vf, uf}}; break;
        case X_minus: xyz = {{pm, vf, -uf}}; break;
        case Y_plus:  xyz = {{uf, pm, vf}}; break;
        case Y_minus: xyz = {{uf, pm, -vf}}; break;
        case Z_plus:  xyz = {{uf, -vf, pm}}; break;
        case Z_minus: xyz = {{uf, vf, pm}}; break;
        default:  xyz = {{0, 0, 0}};
      }
      return xyz;
    }

  private:
    int m_side_length = 0;

  private:
    static UV face_unit_origin(const Face face)
    {
      switch (face)
      {
        case X_plus  : return UV{2, 2};  // Cross shape
        case X_minus : return UV{0, 2};  //
        case Y_plus  : return UV{1, 3};  //      +Y
        case Y_minus : return UV{1, 1};  //  -X  -Z  +X
        case Z_plus  : return UV{1, 0};  //      -Y
        case Z_minus : return UV{1, 2};  //      +Z
        default : return UV{-1, -1};
      }
    }

    static constexpr UV unit_extent()
    {
      return UV{3, 4};
    }
};


class SphericalHarmonics
{
  public:
    SphericalHarmonics(int legendre_order) : m_legendre_order(legendre_order) {}

    // result must be at least size (N+1)^2.
    template <typename T>
    const T* eval_all(const dray::Vec<T, 3> &xyz_normal) const;

    int num_harmonics() const { return num_harmonics(m_legendre_order); }

    static int index(int n, int m) { return n * (n+1) + m; }
    static int alp_index(int n, int m) { return n * (n+1) / 2 + m; }

    static int num_harmonics(int legendre_order)
    {
      return (legendre_order+1)*(legendre_order+1);
    }

  private:
    int m_legendre_order = 0;
    mutable std::vector<char> m_buffer;
};



TEST (dray_spherical_harmonics, dray_cube_map)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "sh_cube_map");
  remove_test_image (output_file);

  const int side_length = 512;
  const CubeMapConverter converter(side_length);
  const CubeMapConverter::UV extent = converter.get_extent();

  dray::Framebuffer frame_buffer(extent.m_u, extent.m_v);
  frame_buffer.clear({{0, 0, 0, 0}});

  const int legendre_order = 2;
  const int n = 2;
  const int m = 2;   // -n <= m <= n

  const SphericalHarmonics spherical_harmonics(legendre_order);

  dray::DeviceFramebuffer dvc_frame_buffer(frame_buffer);
  for (CubeMapConverter::Face face : CubeMapConverter::get_face_list())
  {
    using F = CubeMapConverter::Face;

    for (int v = 0; v < side_length; ++v)
      for (int u = 0; u < side_length; ++u)
      {
        CubeMapConverter::FaceUV faceuv = {face, {u,v}};

        const dray::Vec<float, 3> xyz = converter.to_vec(faceuv).normalized();
        const float * result = spherical_harmonics.eval_all(xyz);
        const float value = result[SphericalHarmonics::index(n,m)];
        const dray::Vec<float, 4> color = {{value, value, value, 1.0f}};

        // For png image
        /// const float avalue = std::abs(value);
        /// const dray::Vec<float, 4> color = {{avalue, avalue, avalue, 1.0f}};

        CubeMapConverter::UV uv = converter.faceuv_to_uv(faceuv);
        dvc_frame_buffer.set_color(uv.m_u + extent.m_u * uv.m_v, color);
      }
  }

  frame_buffer.save(output_file);

  conduit::Node conduit_frame_buffer;
  frame_buffer.to_node(conduit_frame_buffer);
  conduit::relay::io_blueprint::save(conduit_frame_buffer, output_file + ".blueprint_root_hdf5");
}


template <typename T>
const T* SphericalHarmonics::eval_all(const dray::Vec<T, 3> &xyz_normal) const
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

  const int Np1 = m_legendre_order + 1;
  const int Np1_sq = Np1 * Np1;
  const int result_sz = Np1_sq;            // result
  const int sin_sz = Np1;                  // sine
  const int cos_sz = Np1;                  // cosine
  const int alp_sz = Np1 * (Np1+1) / 2;    // associated legendre polynomial
  const int k2_sz = Np1 * (Np1+1) / 2;     // square of normalization constant

  m_buffer.resize(sizeof(T) * (result_sz + sin_sz + cos_sz + alp_sz + k2_sz));

  T * const resultp = (T*) &m_buffer[0];
  T * const sinp = resultp + result_sz;
  T * const cosp = sinp + sin_sz;
  T * const alpp = cosp + cos_sz;
  T * const k2p = alpp + alp_sz;

  const T sqrt2 = std::sqrtl(2);

  const T &x = xyz_normal[0];
  const T &y = xyz_normal[1];
  const T &z = xyz_normal[2];

  // m=0
  {
    const int m = 0;

    sinp[m] = 0;
    cosp[m] = 1;

    // n == m
    alpp[alp_index(m, m)] = 1;
    k2p[alp_index(0, 0)] = 1.0 / (4 * dray::pi());
    resultp[index(m, m)] = std::sqrt(k2p[alp_index(m, m)]) * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(1, 0)] = 2 * (1+1) / (4 * dray::pi());
      resultp[index(m+1, m)] = std::sqrt(k2p[alp_index(m+1, m)]) * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, 0)] = 2 * (n+1) / (4 * dray::pi());

      resultp[index(n, m)] = std::sqrt(k2p[alp_index(n, m)]) * alpp[alp_index(n, m)];
    }
  }

  // m>0
  for (int m = 1; m <= m_legendre_order; ++m)
  {
    sinp[m] = x * sinp[m-1] + y * cosp[m-1];
    cosp[m] = x * cosp[m-1] - y * sinp[m-1];

    // n == m
    alpp[alp_index(m, m)] = (1-2*m) * alpp[alp_index(m-1, m-1)];;
    k2p[alp_index(m, m)] = k2p[alp_index(m-1, m-1)] * (m+1) / (m * (2*m-1) * (2*m));
    resultp[index(m, m)] = std::sqrt(2*k2p[alp_index(m, m)]) * cosp[m] * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(m+1, m)] =
          k2p[alp_index((m+1)-1, m)] * ((m+1)+1) * ((m+1)-m) / ((m+1) * ((m+1)+m));
      resultp[index(m+1, m)] = std::sqrt(2*k2p[alp_index(m+1, m)]) * cosp[m] * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, m)] = k2p[alp_index(n-1, m)] * (n+1) * (n-m) / (n * (n+m));

      resultp[index(n, m)] = std::sqrt(2*k2p[alp_index(n, m)]) * cosp[m] * alpp[alp_index(n, m)];
    }
  }

  // m<0
  for (int m = -1; m >= -m_legendre_order; --m)
  {
    const int absm = -m;
    for (int n = absm; n <= m_legendre_order; ++n)
    {
      resultp[index(n, m)] = std::sqrt(2*k2p[alp_index(n, absm)]) * sinp[absm] * alpp[alp_index(n, absm)];
    }
  }

  return resultp;
}

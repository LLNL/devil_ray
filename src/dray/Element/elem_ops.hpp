// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ELEM_OPS_HPP
#define DRAY_ELEM_OPS_HPP

#include <dray/types.hpp>
#include <dray/Element/element.hpp>
#include <dray/Element/dof_access.hpp>

namespace dray
{

  namespace detail {
    constexpr int32 cartesian_to_tri_idx(int32 i, int32 j, int32 edge)
    {
      // i runs fastest, j slowest.
      // There are a total of (edge)(edge+1)/2 vertices in the triangle.
      // (idx - i) counts the number of vertices below the cap, so
      //
      //   (edge)(edge+1)/2 - (idx - i) = (edge-j)(edge-j+1)/2
      //
      //   j(1 + 2*edge - j)/2 + i = idx

      return (2*edge + 1 - j)*j/2 + i;
    }

    constexpr int32 cartesian_to_tet_idx(int32 i, int32 j, int32 k, int32 e)
    {
      // i runs fastest, k slowest.
      // There are a total of (edge)(edge+1)(edge+2)/6 vertices in the tetrahedron.
      // (idx - cartesian_to_tri_idx(i,j,edge-k)) counts
      // the number of vertices below the cap, so
      //
      //   (edge)(edge+1)(edge+2)/6 - (idx - (2*edge + 1 - j)*j/2 - i)
      //   = (edge-k)(edge-k+1)(edge-k+2)/6
      //
      //   (e)(e+1)(e+2)/6 - (e-k)(e+1-k)(e+2-k)/6 + (2e + 1 - j)*j/2 + i = idx
      //
      //   ((k - 3e - 3)(k) + (3e + 6)e + 2)k/6 + (2e + 1 - j)*j/2 + i = idx

      return (((-1-e)*3+k)*k + (3*e + 6)*e + 2)*k/6 + (2*e + 1 - j)*j/2 + i;
    }
  }


  /// // get_sub_bounds<Simplex>
  /// template <int32 dim, int32 ncomp, int32 P>
  /// DRAY_EXEC void get_sub_bounds(
  ///     const ShapeTAG,                            // Tag for shape
  ///     const OrderPolicy<P> order_p,              // Tag/data for order policy
  ///     WriteDofPtr<Vec<Float, ncomp>> &dof_ptr,                 // dofs read and written here
  ///     const Split<ElemType::Simplex> &split)
  /// {
  ///   //TODO split the triangle element and use coefficients from split element.
  ///   //For now it just uses the non-split coefficients
  ///   //and returns bounds for entire element.

  ///   const int num_dofs = get_num_dofs(ShapeTAG{}, order_p);

  ///
  /// }

  namespace eops
  {

  /** eval_d() */
  template <int32 ncomp>
  DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTri,
                                      OrderPolicy<Linear>,
                                      const ReadDofPtr<Vec<Float, ncomp>> &C,
                                      const Vec<Float, 2> &rc,
                                      Vec<Vec<Float, ncomp>, 2> &out_deriv )
  {
    // C[2]
    // C[0] C[1]
    const Float &u = rc[0], &v = rc[1];
    const Float t = 1.0f - u - v;

    Float sd[3];
    sd[0] = -1.0f;

    // du
    sd[1] = 1.0f;   sd[2] = 0.0f;
    out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

    // du
    sd[1] = 0.0f;   sd[2] = 1.0f;
    out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

    const Float s[3] = { t, u, v };
    return C[0] * s[0] + C[1] * s[1] + C[2] * s[2];
  }

  template <int32 ncomp>
  DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTri,
                                      OrderPolicy<Quadratic>,
                                      const ReadDofPtr<Vec<Float, ncomp>> &C,
                                      const Vec<Float, 2> &rc,
                                      Vec<Vec<Float, ncomp>, 2> &out_deriv )
  {
    // C[6]
    //
    // C[3] C[4]
    //
    // C[0] C[1] C[2]
    const Float &u = rc[0], &v = rc[1];
    const Float t = 1.0f - u - v;

    Float sd[6];
    sd[0] = 2*(-t);

    // -------------------------------
    // du
                      sd[1] = 2*(t-u);    sd[2] = 2*u;
    sd[3] = 2*(-v);   sd[4] = 2*(v);
    sd[5] = 0.0f;
    //
    out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2]
                 + C[3] * sd[3] + C[4] * sd[4]
                 + C[5] * sd[5];
    // -------------------------------

    // -------------------------------
    // dv
                       sd[1] = 2*(-u);   sd[2] = 0.0f;
    sd[3] = 2*(t-v);   sd[4] = 2*(u);
    sd[5] = 2*v;
    //
    out_deriv[1] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2]
                 + C[3] * sd[3] + C[4] * sd[4]
                 + C[5] * sd[5];
    // -------------------------------


    const Float s[6] = { t*t,     2*t*u,   u*u,
                         2*t*v,   2*v*u,
                         v*v };
    return C[0] * s[0] + C[1] * s[1] + C[2] * s[2] +
           C[3] * s[3] + C[4] * s[4] +
           C[5] * s[5];
  }



  template <int32 ncomp>
  DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTet,
                                      OrderPolicy<Linear>,
                                      const ReadDofPtr<Vec<Float, ncomp>> &C,
                                      const Vec<Float, 3> &rc,
                                      Vec<Vec<Float, ncomp>, 3> &out_deriv )
  {
    //  C[2]
    //
    //  C[0]  C[1]
    // C[3]
    const Float &u = rc[0], &v = rc[1], &w = rc[2];
    const Float t = 1.0f - u - v - w;

    Float sd[4];
    sd[0] = -1.0f;

    // du
    sd[1] = 1.0f;   sd[2] = 0.0f;   sd[3] = 0.0f;
    out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

    // du
    sd[1] = 0.0f;   sd[2] = 1.0f;   sd[3] = 0.0f;
    out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

    // dw
    sd[1] = 0.0f;   sd[2] = 0.0f;   sd[3] = 1.0f;
    out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

    const Float s[4] = { t, u, v, w };
    return C[0] * s[0] + C[1] * s[1] + C[2] * s[2] + C[3] * s[3];
  }



  template <int32 ncomp>
  DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTet,
                                      OrderPolicy<Quadratic>,
                                      const ReadDofPtr<Vec<Float, ncomp>> &C,
                                      const Vec<Float, 3> &rc,
                                      Vec<Vec<Float, ncomp>, 3> &out_deriv )
  {
    //  Behold, the P2 tetrahedron
    //
    //              v=1
    //
    //              C[5]
    //             /|   `
    //            / C[3]  C[4]
    //           C[8]        `
    //          /|  C[0]--C[1]--C[2]   u=1
    //         / C[6]   C[7]  '
    //        C[9]   '
    //    w=1
    //
    const Float &u = rc[0], &v = rc[1], &w = rc[2];
    const Float t = 1.0f - u - v - w;

    Float sd[10];
    sd[0] = 2*(-t);

    // -------------------------------
    // du
                      sd[1] = 2*(t-u);   sd[2] = 2*u;
    sd[3] = 2*(-v);   sd[4] = 2*(v);
    sd[5] = 0.0f;
                        sd[6] = 2*(-w);    sd[7] = 2*(w);
                        sd[8] = 0.0f;
                                              sd[9] = 0.0f;
    out_deriv[0] = 0;
    for (int32  i = 0; i < 10; ++i)
      out_deriv[0] += C[i] * sd[i];
    // -------------------------------

    // -------------------------------
    // dv
                       sd[1] = 2*(-u);  sd[2] = 0.0f;
    sd[3] = 2*(t-v);   sd[4] = 2*(u);
    sd[5] = 2*v;
                        sd[6] = 2*(-w);    sd[7] = 0.0f;
                        sd[8] = 2*(w);
                                              sd[9] = 0.0f;
    out_deriv[1] = 0;
    for (int32  i = 0; i < 10; ++i)
      out_deriv[1] += C[i] * sd[i];
    // -------------------------------

    // -------------------------------
    // dv

                     sd[1] = 2*(-u);   sd[2] = 0.0f;
    sd[3] = 2*(-v);  sd[4] = 0.0f;
    sd[5] = 0.0f;
                       sd[6] = 2*(t-w);   sd[7] = 2*(u);
                       sd[8] = 2*(v);
                                            sd[9] = 2*w;
    out_deriv[2] = 0;
    for (int32  i = 0; i < 10; ++i)
      out_deriv[2] += C[i] * sd[i];
    // -------------------------------


    const Float s[10] = { t*t,     2*t*u,   u*u,
                          2*t*v,   2*v*u,
                          v*v,
                                     2*t*w,  2*u*w,
                                     2*v*w,
                                                w*w };

    Vec<Float, ncomp> ret;  ret = 0;
    for (int32 i = 0; i < 10; ++i)
      ret += C[i] * s[i];
    return ret;
  }


  }//eops



  // --------------------------------------------------------------------------
  // split_inplace()
  // --------------------------------------------------------------------------

  namespace detail
  {
    constexpr int32 cartesian_to_tri_idx(int32 i, int32 j, int32 edge);
    constexpr int32 cartesian_to_tet_idx(int32 i, int32 j, int32 k, int32 e);
  }

  // The Split<> object describes a binary split of the simplex at some point
  // (given by 'factor') along an edge (given by vtx_displaced and vtx_tradeoff).
  // Each row of coefficients parallel to the specified edge undergoes
  // 1D-DeCasteljau subdivision. The side closest to the 'tradeoff' vertex is
  // treated as the fixed, exterior node. The side closest to the 'displaced'
  // vertex is treated as the parameter-dependent, interior node.
  //
  //              .                 .           .           .
  //             .-*               . .         . .         . .
  //            .-*-*             . .-*       . . .       . . .
  //           .-*-*-*           . .-*-*     . . .-*     . . . .
  //       (v1=p)    (v0=p)
  //     tradeoff    displaced
  //
  // Subject to axis permutations, the splitting can be carried out as below:
  //
  //   // Triangle
  //   for (0 <= h < p)
  //     for (p-h >= num_updates >= 1)
  //       for (p-h >= v0 > p-h - num_updates, v0+v1 = p-h)
  //         C[v0,v1;h] := f*C[v0,v1;h] + (1-f)*C[v0-1,v1+1;h];
  //
  //   // Tetrahedron
  //   for (0 <= g+h < p)
  //     for (p-(g+h) >= num_updates >= 1)
  //       for (p-(g+h) >= v0 > p-(g+h) - num_updates, v0+v1 = p-(g+h))
  //         C[v0,v1;g,h] := f*C[v0,v1;g,h] + (1-f)*C[v0-1,v1+1;g,h];
  //

  //
  // split_inplace<2, Simplex>        (Triangle)
  //
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<2, ncomp, ElemType::Simplex, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Simplex> &split)
  {
    const int8 p = (int8) elem_info.get_order();
    const int8 v0 = (int8) split.vtx_displaced;
    const int8 v1 = (int8) split.vtx_tradeoff;
    const int8 v2 = 0+1+2 - v0 - v1;

    int8 b[3];  // barycentric indexing

    // I think this way of expressing the permuation is most readable.
    // On the other hand, potential for loop unrolling might be easier to
    // detect if the permutation was expressed using the inverse.

    for (b[v2] = 0; b[v2] < p; ++b[v2])
      for (int8 num_updates = p-b[v2]; num_updates >= 1; --num_updates)
        for (b[v0] = p-b[v2]; b[v0] > p-b[v2] - num_updates; --b[v0])
        {
          b[v1] = p-b[v2]-b[v0];

          int8 b_left[3];
          b_left[v0] = b[v0] - 1;
          b_left[v1] = b[v1] + 1;
          b_left[v2] = b[v2];

          const int32 right = detail::cartesian_to_tri_idx(b[0], b[1], p+1);
          const int32 left = detail::cartesian_to_tri_idx(b_left[0], b_left[1], p+1);

          dof_ptr[right] =
              dof_ptr[right] * split.factor + dof_ptr[left] * (1-split.factor);
        }
  }


  //
  // split_inplace<3, Simplex>          (Tetrahedron)
  //
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<3, ncomp, ElemType::Simplex, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Simplex> &split)
  {
    const int8 p = (int8) elem_info.get_order();
    const int8 v0 = (int8) split.vtx_displaced;
    const int8 v1 = (int8) split.vtx_tradeoff;

    const uint8 avail = -1u & ~(1u << v0) & ~(1u << v1);
    const int8 v2 = (avail & 1u) ? 0 : (avail & 2u) ? 1 : (avail & 4u) ? 2 : 3;
    const int8 v3 = 0+1+2+3 - v0 - v1 - v2;

    int8 b[4];  // barycentric indexing

    for (b[v3] = 0; b[v3] < p; ++b[v3])
      for (b[v2] = 0; b[v2] < p-b[v3]; ++b[v2])
      {
        const int8 gph = b[v2] + b[v3];

        for (int8 num_updates = p-gph; num_updates >= 1; --num_updates)
          for (b[v0] = p-gph; b[v0] > p-gph - num_updates; --b[v0])
          {
            b[v1] = p-gph-b[v0];

            int8 b_left[4];
            b_left[v0] = b[v0] - 1;
            b_left[v1] = b[v1] + 1;
            b_left[v2] = b[v2];
            b_left[v3] = b[v3];

            const int32 right = detail::cartesian_to_tet_idx(
                b[0], b[1], b[2], p+1);
            const int32 left = detail::cartesian_to_tet_idx(
                b_left[0], b_left[1], b_left[2], p+1);

            dof_ptr[right] =
                dof_ptr[right] * split.factor + dof_ptr[left] * (1-split.factor);
          }
      }
  }

  // Binary split on quad:
  //
  //  left:
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //
  //  right:
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //

  //
  // split_inplace<Tensor>
  //
  template <int32 dim, int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<dim, ncomp, ElemType::Tensor, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Tensor> &split)
  {
    const uint32 p = elem_info.get_order();

    uint32 p_order_pow[4];
    p_order_pow[0] = 1;
    p_order_pow[1] = p_order_pow[0] * (p + 1);
    p_order_pow[2] = p_order_pow[1] * (p + 1);
    p_order_pow[3] = p_order_pow[2] * (p + 1);

    const int32 &axis = split.axis;
    static_assert((1 <= dim && dim <= 3), "split_inplace() only supports 1D, 2D, or 3D.");
    assert((0 <= axis && axis < dim));
    const uint32 stride = p_order_pow[axis];
    const uint32 chunk_sz = p_order_pow[axis+1];
    const uint32 num_chunks = p_order_pow[dim - (axis+1)];

    if (!split.f_lower_t_upper)
    {
      // Left
      for (int32 chunk = 0; chunk < num_chunks; ++chunk, dof_ptr += chunk_sz)
      {
        // Split the chunk along axis.
        // If there are axes below axis, treat them as a vector of dofs.

        // In DeCasteljau left split, we repeatedly overwrite the right side.
        for (int32 from_front = 1; from_front <= p; ++from_front)
          for (int32 ii = p; ii >= 0+from_front; --ii)
            for (int32 e = 0; e < stride; ++e)
            {
              dof_ptr[ii*stride + e] =
                  dof_ptr[(ii-1)*stride + e] * (1 - split.factor)
                  + dof_ptr[ii*stride + e] * (split.factor);
            }
      }
    }
    else
    {
      // Right
      for (int32 chunk = 0; chunk < num_chunks; ++chunk, dof_ptr += chunk_sz)
      {
        // Split the chunk along axis.
        // If there are axes below axis, treat them as a vector of dofs.

        // In DeCasteljau right split, we repeatedly overwrite the left side.
        for (int32 from_back = 1; from_back <= p; ++from_back)
          for (int32 ii = 0; ii <= p-from_back; ++ii)
            for (int32 e = 0; e < stride; ++e)
            {
              dof_ptr[ii*stride + e] =
                  dof_ptr[ii*stride + e] * (1 - split.factor)
                  + dof_ptr[(ii+1)*stride + e] * (split.factor);
            }
      }
    }
  }

  // --------------------------------------------------------------------------



  // Imported from isosurface_meshing

/// namespace DeCasteljauSplitting
/// {
/// // sub_element()
/// //
/// // Replaces sub_element_fixed_order()
/// // This version operates in-place and does not assume fixed order.
/// template <uint32 dim, uint32 ncomp>
/// DRAY_EXEC void sub_element(uint32 p_order,
///                            const Range *ref_boxs,
///                            WriteDofPtr<Vec<Float, ncomp>> &wptr)
/// {
///   // Split along each axis sequentially. It is a tensor element.
///   for (int32 d = 0; d < dim; ++d)
///   {
///     const Float t1 = ref_boxs[d].max();
///     Float t0 = ref_boxs[d].min();
///
///     // Split left, using right endpoint.
///     if (t1 < 1.0)
///       split_inplace_left(wptr, t1, dim, d, p_order);
///
///     // Left endpoint relative to the new subinterval.
///     if (t1 > 0.0) t0 /= t1;
///
///     // Split right, using left endpoint.
///     if (t0 > 0.0)
///       split_inplace_right(wptr, t0, dim, d, p_order);
///   }
/// }





}//namespace dray

#endif//DRAY_ELEM_OPS_HPP

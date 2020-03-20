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
  // split_inplace<Tri>
  template <uint32 dim, uint32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<dim, ncomp, ElemType::Tri, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> &dof_ptr,
      const Split<ElemType::Tri> &split)
  {
    // TODO NOW
  }

  // split_inplace<Quad>
  template <uint32 dim, uint32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<dim, ncomp, ElemType::Quad, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> &dof_ptr,
      const Split<ElemType::Quad> &split)
  {
    // TODO NOW
  }








  // Imported from isosurface_meshing

/// namespace DeCasteljauSplitting
/// {
///   // split_inplace_left()
///   //
///   // 1D left split operator in a single ref axis, assuming tensorized element.
///   template <uint32 ncomp>
///   DRAY_EXEC void split_inplace_left(WriteDofPtr<Vec<Float, ncomp>> &wptr, Float t1, uint32 dim, uint32 axis, uint32 p_order)
///   {
///     uint32 p_order_pow[4];
///     p_order_pow[0] = 1;
///     p_order_pow[1] = p_order_pow[0] * (p_order + 1);
///     p_order_pow[2] = p_order_pow[1] * (p_order + 1);
///     p_order_pow[3] = p_order_pow[2] * (p_order + 1);
/// 
///     assert(dim <= 3);
///     assert(axis < dim);
///     const uint32 stride = p_order_pow[axis];
///     const uint32 chunk_sz = p_order_pow[axis+1];
///     const uint32 num_chunks = p_order_pow[dim - (axis+1)];
/// 
///     const Float left = 1.0f - t1;
///     const Float &right = t1;
///     const uint32 & p = p_order;
/// 
///     for (int32 chunk = 0; chunk < num_chunks; ++chunk, wptr += chunk_sz)
///     {
///       // Split the chunk along axis.
///       // If there are axes below axis, treat them as a vector of dofs.
/// 
///       // In DeCasteljau left split, we repeatedly overwrite the right side.
///       for (int32 from_front = 1; from_front <= p; ++from_front)
///         for (int32 ii = p; ii >= 0+from_front; --ii)
///           for (int32 e = 0; e < stride; ++e)
///           {
///             wptr[ii*stride + e] = wptr[(ii-1)*stride + e] * left
///                                     + wptr[ii*stride + e] * right;
///           }
///     }
///   }
/// 
///   // split_inplace_right()
///   //
///   // 1D right split operator in a single ref axis, assuming tensorized element.
///   template <uint32 ncomp>
///   DRAY_EXEC void split_inplace_right(WriteDofPtr<Vec<Float, ncomp>> &wptr, Float t0, uint32 dim, uint32 axis, uint32 p_order)
///   {
///     uint32 p_order_pow[4];
///     p_order_pow[0] = 1;
///     p_order_pow[1] = p_order_pow[0] * (p_order + 1);
///     p_order_pow[2] = p_order_pow[1] * (p_order + 1);
///     p_order_pow[3] = p_order_pow[2] * (p_order + 1);
/// 
///     assert(dim <= 3);
///     assert(axis < dim);
///     const uint32 stride = p_order_pow[axis];
///     const uint32 chunk_sz = p_order_pow[axis+1];
///     const uint32 num_chunks = p_order_pow[dim - (axis+1)];
/// 
///     const Float left = 1.0f - t0;
///     const Float &right = t0;
///     const uint32 & p = p_order;
/// 
///     for (int32 chunk = 0; chunk < num_chunks; ++chunk, wptr += chunk_sz)
///     {
///       // Split the chunk along axis.
///       // If there are axes below axis, treat them as a vector of dofs.
/// 
///       // In DeCasteljau right split, we repeatedly overwrite the left side.
///       for (int32 from_back = 1; from_back <= p; ++from_back)
///         for (int32 ii = 0; ii <= p-from_back; ++ii)
///           for (int32 e = 0; e < stride; ++e)
///           {
///             wptr[ii*stride + e] =       wptr[ii*stride + e] * left
///                                   + wptr[(ii+1)*stride + e] * right;
///           }
///     }
///   }
/// 
/// };
/// 
/// 
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

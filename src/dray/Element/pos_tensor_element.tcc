// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POS_TENSOR_ELEMENT_HPP
#define DRAY_POS_TENSOR_ELEMENT_HPP

/**
 * @file pos_tensor_element.hpp
 * @brief Partial template specialization of Element_impl
 *        for tensor (i.e. hex and quad) elements.
 */

#include <dray/Element/element.hpp>
#include <dray/integer_utils.hpp> // MultinomialCoeff
#include <dray/vec.hpp>

#include <dray/Element/bernstein_basis.hpp> // get_sub_coefficient

namespace dray
{

template <uint32 dim> class QuadRefSpace
{
  public:
  DRAY_EXEC static bool is_inside (const Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static void clamp_to_domain (Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static Vec<Float, dim>
  project_to_domain (const Vec<Float, dim> &r1, const Vec<Float, dim> &r2); // TODO
};


// Specialize SubRef for Quad type.
template <> struct ElemTypeAttributes<ElemType::Quad>
{
  template <uint32 dim> using SubRef = AABB<dim>;
};

// TODO add get_sub_bounds to each specialization.

// ---------------------------------------------------------------------------


// -----
// Implementations
// -----

template <uint32 dim>
DRAY_EXEC bool QuadRefSpace<dim>::is_inside (const Vec<Float, dim> &ref_coords)
{
  Float min_val = 2.f;
  Float max_val = -1.f;
  for (int32 d = 0; d < dim; d++)
  {
    min_val = min (ref_coords[d], min_val);
    max_val = max (ref_coords[d], max_val);
  }
  return (min_val >= 0.f - epsilon<Float> ()) && (max_val <= 1.f + epsilon<Float> ());
}

template <uint32 dim>
DRAY_EXEC void QuadRefSpace<dim>::clamp_to_domain (Vec<Float, dim> &ref_coords)
{
  // TODO
}

template <uint32 dim>
DRAY_EXEC Vec<Float, dim>
QuadRefSpace<dim>::project_to_domain (const Vec<Float, dim> &r1, const Vec<Float, dim> &r2)
{
  return { 0.0 }; // TODO
}


// ---------------------------------------------------------------------------

// Template specialization (Quad type, general order).
//
// Assume dim <= 3.
//
template <uint32 dim, uint32 ncomp>
class Element_impl<dim, ncomp, ElemType::Quad, Order::General> : public QuadRefSpace<dim>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;
  uint32 m_order;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 poly_order)
  {
    m_dof_ptr = dof_ptr;
    m_order = poly_order;
  }
  DRAY_EXEC int32 get_order () const
  {
    return m_order;
  }
  DRAY_EXEC int32 get_num_dofs () const
  {
    return get_num_dofs (m_order);
  }
  DRAY_EXEC static constexpr int32 get_num_dofs (int32 order)
  {
    return intPow (order + 1, dim);
  }

  //DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, dim> &r) const
  //{
  //  using DofT = Vec<Float, ncomp>;
  //  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  //  // TODO
  //  DofT answer;
  //  answer = 0;
  //  return answer;
  //}

  DRAY_EXEC Vec<Float, ncomp> eval_d (const Vec<Float, dim> &ref_coords,
                                      Vec<Vec<Float, ncomp>, dim> &out_derivs) const
  {
    // Directly evaluate a Bernstein polynomial with a hybrid of Horner's rule and accumulation of powers:
    //     V = 0.0;  xpow = 1.0;
    //     for(i)
    //     {
    //       V = V*(1-x) + C[i]*xpow*nchoosek(p,i);
    //       xpow *= x;
    //     }
    //
    // Indirectly evaluate a high-order Bernstein polynomial, by directly evaluating
    // the two parent lower-order Bernstein polynomials, and mixing with weights {(1-x), x}.
    //
    // Indirectly evaluate the derivative of a high-order Bernstein polynomial, by directly
    // evaluating the two parent lower-order Bernstein polynomials, and mixing with weights {-p, p}.

    using DofT = Vec<Float, ncomp>;
    using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

    DofT zero;
    zero = 0;

    const Float u = (dim > 0 ? ref_coords[0] : 0.0);
    const Float v = (dim > 1 ? ref_coords[1] : 0.0);
    const Float w = (dim > 2 ? ref_coords[2] : 0.0);
    const Float ubar = 1.0 - u;
    const Float vbar = 1.0 - v;
    const Float wbar = 1.0 - w;

    const int32 p1 = (dim >= 1 ? m_order : 0);
    const int32 p2 = (dim >= 2 ? m_order : 0);
    const int32 p3 = (dim >= 3 ? m_order : 0);

    int32 B[MaxPolyOrder];
    if (m_order >= 1)
    {
      BinomialCoeff binomial_coeff;
      binomial_coeff.construct (m_order - 1);
      for (int32 ii = 0; ii <= m_order - 1; ii++)
      {
        B[ii] = binomial_coeff.get_val ();
        binomial_coeff.slide_over (0);
      }
    }

    int32 cidx = 0; // Index into element dof indexing space.

    // Compute and combine order (p-1) values to get order (p) values/derivatives.
    // https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties

    DofT val_u, val_v, val_w;
    DofT deriv_u;
    Vec<DofT, 2> deriv_uv;
    Vec<DofT, 3> deriv_uvw;

    // Level3 set up.
    Float wpow = 1.0;
    Vec<DofT, 3> val_w_L, val_w_R; // Second/third columns are derivatives in lower level.
    val_w_L = zero;
    val_w_R = zero;
    for (int32 ii = 0; ii <= p3; ii++)
    {
      // Level2 set up.
      Float vpow = 1.0;
      Vec<DofT, 2> val_v_L, val_v_R; // Second column is derivative in lower level.
      val_v_L = zero;
      val_v_R = zero;

      for (int32 jj = 0; jj <= p2; jj++)
      {
        // Level1 set up.
        Float upow = 1.0;
        DofT val_u_L = zero, val_u_R = zero; // L and R can be combined --> val, deriv.
        DofT C = m_dof_ptr[cidx++];
        for (int32 kk = 1; kk <= p1; kk++)
        {
          // Level1 accumulation.
          val_u_L = val_u_L * ubar + C * (B[kk - 1] * upow);
          C = m_dof_ptr[cidx++];
          val_u_R = val_u_R * ubar + C * (B[kk - 1] * upow);
          upow *= u;
        } // kk

        // Level1 result.
        val_u = (p1 > 0 ? val_u_L * ubar + val_u_R * u : C);
        deriv_u = (val_u_R - val_u_L) * p1;

        // Level2 accumulation.
        if (jj > 0)
        {
          val_v_R[0] = val_v_R[0] * vbar + val_u * (B[jj - 1] * vpow);
          val_v_R[1] = val_v_R[1] * vbar + deriv_u * (B[jj - 1] * vpow);
          vpow *= v;
        }
        if (jj < p2)
        {
          val_v_L[0] = val_v_L[0] * vbar + val_u * (B[jj] * vpow);
          val_v_L[1] = val_v_L[1] * vbar + deriv_u * (B[jj] * vpow);
        }
      } // jj

      // Level2 result.
      val_v = (p2 > 0 ? val_v_L[0] * vbar + val_v_R[0] * v : val_u);
      deriv_uv[0] = (p2 > 0 ? val_v_L[1] * vbar + val_v_R[1] * v : deriv_u);
      deriv_uv[1] = (val_v_R[0] - val_v_L[0]) * p2;

      // Level3 accumulation.
      if (ii > 0)
      {
        val_w_R[0] = val_w_R[0] * wbar + val_v * (B[ii - 1] * wpow);
        val_w_R[1] = val_w_R[1] * wbar + deriv_uv[0] * (B[ii - 1] * wpow);
        val_w_R[2] = val_w_R[2] * wbar + deriv_uv[1] * (B[ii - 1] * wpow);
        wpow *= w;
      }
      if (ii < p3)
      {
        val_w_L[0] = val_w_L[0] * wbar + val_v * (B[ii] * wpow);
        val_w_L[1] = val_w_L[1] * wbar + deriv_uv[0] * (B[ii] * wpow);
        val_w_L[2] = val_w_L[2] * wbar + deriv_uv[1] * (B[ii] * wpow);
      }
    } // ii

    // Level3 result.
    val_w = (p3 > 0 ? val_w_L[0] * wbar + val_w_R[0] * w : val_v);
    deriv_uvw[0] = (p3 > 0 ? val_w_L[1] * wbar + val_w_R[1] * w : deriv_uv[0]);
    deriv_uvw[1] = (p3 > 0 ? val_w_L[2] * wbar + val_w_R[2] * w : deriv_uv[1]);
    deriv_uvw[2] = (val_w_R[0] - val_w_L[0]) * p3;

    if (dim > 0) out_derivs[0] = deriv_uvw[0];
    if (dim > 1) out_derivs[1] = deriv_uvw[1];
    if (dim > 2) out_derivs[2] = deriv_uvw[2];

    return val_w;
  }

  DRAY_EXEC void get_sub_bounds (const AABB<dim> &sub_ref, AABB<ncomp> &aabb) const;

  // Multivariate gradient is a Jacobian matrix. For consistency with column major indexing,
  // index as: physical component moving fastest and reference axis moving slowest.
  DRAY_EXEC void bound_grad_aabb(AABB<dim*ncomp> &grad_bounds) const;

  // In this case the lower bound is not actually a lower bound
  // on the function, just the coefficients. Upper bound is valid though.
  DRAY_EXEC void bound_grad_mag2(Range<> &mag2_range) const;

  // Multivariate Hessian is a 3rd order tensor.
  // Index as: physical component moving fastest and reference axes moving slowest.
  DRAY_EXEC void bound_hess_aabb(AABB<dim*dim*ncomp> &hess_bounds) const;

  // In this case the lower bound is not actually a lower bound
  // on the function, just the coefficients. Upper bound is valid though.
  DRAY_EXEC void bound_hess_mag2(Range<> &mag2_range) const;

  // Project element into higher order basis, e.g. for better bounds.
  // Technically it's not a projection but an inclusion.
  // The polynomial should be the same, just represented differently.
  // assert(raise == hi_elem.get_order() - lo_elem.get_order());
  //TODO make a writeable element class and use that for hi_elem, delete hi_coeffs.
  template <uint32 raise>
  static DRAY_EXEC void project_to_higher_order_basis(const Element_impl &lo_elem,
                                                      Element_impl &hi_elem,
                                                      WriteDofPtr<Vec<Float, ncomp>> &hi_coeffs);
};


//
// get_sub_bounds()
template <uint32 dim, uint32 ncomp>
DRAY_EXEC void
Element_impl<dim, ncomp, ElemType::Quad, Order::General>::get_sub_bounds (const AABB<dim> &sub_ref,
                                                                          AABB<ncomp> &aabb) const
{
  // Initialize.
  aabb.reset ();

  const int32 num_dofs = get_num_dofs ();

  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  if (m_order <= 3) // TODO find the optimal threshold, if there is one.
  {
    // Get the sub-coefficients all at once in a block.
    switch (m_order)
    {
    case 1:
    {
      constexpr int32 POrder = 1;
      MultiVec<Float, dim, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<dim, ncomp, POrder, PtrT> (sub_ref.m_ranges, m_dof_ptr);
      for (int32 ii = 0; ii < num_dofs; ii++)
        aabb.include (sub_nodes.linear_idx (ii));
    }
    break;

    case 2:
    {
      constexpr int32 POrder = 2;
      MultiVec<Float, dim, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<dim, ncomp, POrder, PtrT> (sub_ref.m_ranges, m_dof_ptr);
      for (int32 ii = 0; ii < num_dofs; ii++)
        aabb.include (sub_nodes.linear_idx (ii));
    }
    break;

    case 3:
    {
      constexpr int32 POrder = 3;
      MultiVec<Float, dim, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<dim, ncomp, POrder, PtrT> (sub_ref.m_ranges, m_dof_ptr);
      for (int32 ii = 0; ii < num_dofs; ii++)
        aabb.include (sub_nodes.linear_idx (ii));
    }
    break;
    }
  }
  else
  {
    // Get each sub-coefficient one at a time.
    for (int32 i0 = 0; i0 <= (dim >= 1 ? m_order : 0); i0++)
      for (int32 i1 = 0; i1 <= (dim >= 2 ? m_order : 0); i1++)
        for (int32 i2 = 0; i2 <= (dim >= 3 ? m_order : 0); i2++)
        {
          Vec<Float, ncomp> sub_node =
          // TODO move out of bernstein_basis.hpp
          BernsteinBasis<dim>::template get_sub_coefficient<PtrT, ncomp> (
          sub_ref.m_ranges, m_dof_ptr, m_order, i0, i1, i2);
          aabb.include (sub_node);
        }
  }
}

template <uint32 ncomp, typename PtrT>
struct SameDegreeDerivCoefficients
{
  //TODO refactor the if statements, which ultimately are called in inner loops.

  static void DRAY_EXEC store_D1(
      const PtrT &dof_ptr,
      int32 cidx,
      int32 offset,
      int32 p,
      int32 i,
      Vec<Float, ncomp> &coeff)
  {
    // Using the degree-(p, p, p) Bernstin basis for all partial derivatives,
    // so we can apply Bernstein bounds.
    //
    //   partial_0[i,j,k]
    //   = (p-i)C[i+1,j,k] + (2i-p)C[i,j,k] - iC[i-1,j,k]
    //
    coeff = dof_ptr[cidx] * (2*i - p);
    if (i > 0)
      coeff += dof_ptr[cidx - offset] * (-i);
    if (i < p)
      coeff += dof_ptr[cidx + offset] * (p-i);
  }

  static void DRAY_EXEC store_D2_diag(
      const PtrT &dof_ptr,
      int32 cidx,
      int32 offset,
      int32 p,
      int32 i,
      Vec<Float, ncomp> &coeff)
  {
    // partial_00[i,j,k]
    coeff = dof_ptr[cidx] * (p*p - p - 6*i*p + 6*i*i);
    if (i > 1)   coeff += dof_ptr[cidx - 2*offset] * (i*(i-1));
    if (i > 0)   coeff += dof_ptr[cidx -   offset] * ((-i)*(4*i - 2*p + 2));
    if (i < p)   coeff += dof_ptr[cidx +   offset] * ((p-i)*(4*i - 2*p + 2));
    if (i < p-1) coeff += dof_ptr[cidx + 2*offset] * ((p-i)*(p-i-1));
  }

  static void DRAY_EXEC store_D2_off(
      const PtrT &dof_ptr,
      int32 cidx,
      int32 si,  int32 sj,
      int32 pi,  int32 pj,
      int32 i,   int32 j,
      Vec<Float, ncomp> &coeff)
  {
    const int32 bi0 = -i,  bi1 = (2*i - pi),  bi2 = (pi - i);
    const int32 bj0 = -j,  bj1 = (2*j - pj),  bj2 = (pj - j);

    coeff = dof_ptr[cidx] * (bj1 * bi1);

    if (j > 0)
    {
      if (i > 0)   coeff += dof_ptr[cidx - si - sj] * (bj0 * bi0);
      if (true)    coeff += dof_ptr[cidx      - sj] * (bj0 * bi1);
      if (i < pi)  coeff += dof_ptr[cidx + si - sj] * (bj0 * bi2);
    }

    if (i > 0)     coeff += dof_ptr[cidx - si     ] * (bj1 * bi0);

    if (i < pi)    coeff += dof_ptr[cidx + si     ] * (bj1 * bi2);

    if (j < pj)
    {
      if (i > 0)   coeff += dof_ptr[cidx - si + sj] * (bj2 * bi0);
      if (true)    coeff += dof_ptr[cidx      + sj] * (bj2 * bi1);
      if (i < pi)  coeff += dof_ptr[cidx + si + sj] * (bj2 * bi2);
    }
  }
};


//
// bound_grad_aabb()
template <uint32 dim, uint32 ncomp>
DRAY_EXEC void
Element_impl<dim, ncomp, ElemType::Quad, Order::General>::bound_grad_aabb(
    AABB<dim*ncomp> &grad_bounds) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  grad_bounds.reset();

  // Two interfaces to the same matrix/vector.
  Vec<Float, dim * ncomp> pd_coeff_data;
  Vec<DofT, dim> &pd_coeff_vvec = *(Vec<DofT, dim> *) &pd_coeff_data;

  // Note these are named 1..3 in eval() but 0..2 sounds better.
  const int32 p0 = (dim >= 1 ? m_order : 0);
  const int32 p1 = (dim >= 2 ? m_order : 0);
  const int32 p2 = (dim >= 3 ? m_order : 0);
  const int32 s0 = 1;
  const int32 s1 = s0 * (p0+1);
  const int32 s2 = s1 * (p1+1);

  using SDDC = SameDegreeDerivCoefficients<ncomp, PtrT>;

  int32 cidx = 0; // Index into element dof indexing space.

  for (int32 i_ = 0; i_ <= p2; ++i_)
    for (int32 j_ = 0; j_ <= p1; ++j_)
      for (int32 k_ = 0; k_ <= p0; ++k_)  // x moves fastest.
      {
        // partial x coefficient
        if (dim >= 1)
          SDDC::store_D1(m_dof_ptr, cidx, s0, p0, k_, pd_coeff_vvec[0]);

        // partial y coefficient
        if (dim >= 2)
          SDDC::store_D1(m_dof_ptr, cidx, s1, p1, j_, pd_coeff_vvec[1]);

        // partial z coefficient
        if (dim >= 3)
          SDDC::store_D1(m_dof_ptr, cidx, s2, p2, i_, pd_coeff_vvec[2]);

        grad_bounds.include(pd_coeff_data);
        ++cidx;
      }
}


//
// bound_grad_mag2()
template <uint32 dim, uint32 ncomp>
DRAY_EXEC void
Element_impl<dim, ncomp, ElemType::Quad, Order::General>::bound_grad_mag2(
    Range<> &mag2_range) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  mag2_range.reset();

  // Two interfaces to the same matrix/vector.
  Vec<Float, dim * ncomp> pd_coeff_data;
  Vec<DofT, dim> &pd_coeff_vvec = *(Vec<DofT, dim> *) &pd_coeff_data;

  // Note these are named 1..3 in eval() but 0..2 sounds better.
  const int32 p0 = (dim >= 1 ? m_order : 0);
  const int32 p1 = (dim >= 2 ? m_order : 0);
  const int32 p2 = (dim >= 3 ? m_order : 0);
  const int32 s0 = 1;
  const int32 s1 = s0 * (p0+1);
  const int32 s2 = s1 * (p1+1);

  using SDDC = SameDegreeDerivCoefficients<ncomp, PtrT>;

  int32 cidx = 0; // Index into element dof indexing space.

  for (int32 i_ = 0; i_ <= p2; ++i_)
    for (int32 j_ = 0; j_ <= p1; ++j_)
      for (int32 k_ = 0; k_ <= p0; ++k_)  // x moves fastest.
      {
        // partial x coefficient
        if (dim >= 1)
          SDDC::store_D1(m_dof_ptr, cidx, s0, p0, k_, pd_coeff_vvec[0]);

        // partial y coefficient
        if (dim >= 2)
          SDDC::store_D1(m_dof_ptr, cidx, s1, p1, j_, pd_coeff_vvec[1]);

        // partial z coefficient
        if (dim >= 3)
          SDDC::store_D1(m_dof_ptr, cidx, s2, p2, i_, pd_coeff_vvec[2]);

        mag2_range.include(pd_coeff_data.magnitude2());
        ++cidx;
      }
}

//
// bound_hess_aabb()
template <uint32 dim, uint32 ncomp>
DRAY_EXEC void
Element_impl<dim, ncomp, ElemType::Quad, Order::General>::bound_hess_aabb(
    AABB<dim*dim*ncomp> &hess_bounds) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  hess_bounds.reset();

  // Two interfaces to the same tensor/matrix.
  Vec<Float, dim * dim * ncomp> pdd_coeff_data;
  Vec<Vec<DofT, dim>, dim> &pdd_coeff_vvvec =
        *(Vec<Vec<DofT, dim>, dim> *) &pdd_coeff_data;
  //things are getting crazy.. Vec<Vec<Vec<>>>

  // Note these are named 1..3 in eval() but 0..2 sounds better.
  const int32 p0 = (dim >= 1 ? m_order : 0);
  const int32 p1 = (dim >= 2 ? m_order : 0);
  const int32 p2 = (dim >= 3 ? m_order : 0);
  const int32 s0 = 1;
  const int32 s1 = s0 * (p0+1);
  const int32 s2 = s1 * (p1+1);

  using SDDC = SameDegreeDerivCoefficients<ncomp, PtrT>;

  int32 cidx = 0; // Index into element dof indexing space.

  for (int32 i_ = 0; i_ <= p2; ++i_)
    for (int32 j_ = 0; j_ <= p1; ++j_)
      for (int32 k_ = 0; k_ <= p0; ++k_)  // x moves fastest.
      {
        // Grow out the matrix by shells:   x   y   z
        //                                      |   |
        //                                 (y)--y   z
        //                                          |
        //                                 (z)-(z)--z

        if (dim >= 1)
          SDDC::store_D2_diag(m_dof_ptr, cidx, s0, p0, k_, pdd_coeff_vvvec[0][0]);

        if (dim >= 2)
        {
          SDDC::store_D2_off(
              m_dof_ptr, cidx, s0, s1, p0, p1, k_, j_, pdd_coeff_vvvec[0][1]);
          SDDC::store_D2_diag(
              m_dof_ptr, cidx, s1, p1, j_, pdd_coeff_vvvec[1][1]);

          // Symmetry by Clairaut's theorem.
          pdd_coeff_vvvec[1][0] = pdd_coeff_vvvec[0][1];
        }
 
        if (dim >= 3)
        {
          SDDC::store_D2_off(
              m_dof_ptr, cidx, s0, s2, p0, p2, k_, i_, pdd_coeff_vvvec[0][2]);
          SDDC::store_D2_off(
              m_dof_ptr, cidx, s1, s2, p1, p2, j_, i_, pdd_coeff_vvvec[1][2]);
          SDDC::store_D2_diag(
              m_dof_ptr, cidx, s2, p2, i_, pdd_coeff_vvvec[2][2]);

          // Symmetry by Clairaut's theorem.
          pdd_coeff_vvvec[2][1] = pdd_coeff_vvvec[1][2];
          pdd_coeff_vvvec[2][0] = pdd_coeff_vvvec[0][2];
        }

        hess_bounds.include(pdd_coeff_data);
        ++cidx;
      }
}

//
// bound_hess_mag2()
template <uint32 dim, uint32 ncomp>
DRAY_EXEC void
Element_impl<dim, ncomp, ElemType::Quad, Order::General>::bound_hess_mag2(
    Range<> &mag2_range) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  mag2_range.reset();

  // Two interfaces to the same tensor/matrix.
  Vec<Float, dim * dim * ncomp> pdd_coeff_data;
  Vec<Vec<DofT, dim>, dim> &pdd_coeff_vvvec =
        *(Vec<Vec<DofT, dim>, dim> *) &pdd_coeff_data;
  //things are getting crazy.. Vec<Vec<Vec<>>>

  // Note these are named 1..3 in eval() but 0..2 sounds better.
  const int32 p0 = (dim >= 1 ? m_order : 0);
  const int32 p1 = (dim >= 2 ? m_order : 0);
  const int32 p2 = (dim >= 3 ? m_order : 0);
  const int32 s0 = 1;
  const int32 s1 = s0 * (p0+1);
  const int32 s2 = s1 * (p1+1);

  using SDDC = SameDegreeDerivCoefficients<ncomp, PtrT>;

  int32 cidx = 0; // Index into element dof indexing space.

  for (int32 i_ = 0; i_ <= p2; ++i_)
    for (int32 j_ = 0; j_ <= p1; ++j_)
      for (int32 k_ = 0; k_ <= p0; ++k_)  // x moves fastest.
      {
        // Grow out the matrix by shells:   x   y   z
        //                                      |   |
        //                                 (y)--y   z
        //                                          |
        //                                 (z)-(z)--z

        if (dim >= 1)
          SDDC::store_D2_diag(m_dof_ptr, cidx, s0, p0, k_, pdd_coeff_vvvec[0][0]);

        if (dim >= 2)
        {
          SDDC::store_D2_off(
              m_dof_ptr, cidx, s0, s1, p0, p1, k_, j_, pdd_coeff_vvvec[0][1]);
          SDDC::store_D2_diag(
              m_dof_ptr, cidx, s1, p1, j_, pdd_coeff_vvvec[1][1]);

          // Symmetry by Clairaut's theorem.
          pdd_coeff_vvvec[1][0] = pdd_coeff_vvvec[0][1];
        }
 
        if (dim >= 3)
        {
          SDDC::store_D2_off(
              m_dof_ptr, cidx, s0, s2, p0, p2, k_, i_, pdd_coeff_vvvec[0][2]);
          SDDC::store_D2_off(
              m_dof_ptr, cidx, s1, s2, p1, p2, j_, i_, pdd_coeff_vvvec[1][2]);
          SDDC::store_D2_diag(
              m_dof_ptr, cidx, s2, p2, i_, pdd_coeff_vvvec[2][2]);

          // Symmetry by Clairaut's theorem.
          pdd_coeff_vvvec[2][1] = pdd_coeff_vvvec[1][2];
          pdd_coeff_vvvec[2][0] = pdd_coeff_vvvec[0][2];
        }

        mag2_range.include(pdd_coeff_data.magnitude2());
        ++cidx;
      }
}


template <uint32 dim, uint32 ncomp>
template <uint32 raise>
DRAY_EXEC void
Element_impl<dim, ncomp, ElemType::Quad, Order::General>::
project_to_higher_order_basis(const Element_impl &lo_elem,
                              Element_impl &hi_elem,
                              WriteDofPtr<Vec<Float, ncomp>> &hi_coeffs)
{
  // Formula to raise from order (p-r) to order (p):
  //
  //   C^p_i = \frac{(p-r)!}{p!} \sum_{s=0}^r {r \choose s} \frac{i!}{(i-r+s)!} \frac{(p-i)!}{(p-i-s)!} C^{p-r}_{i-r+s}
  //
  // Terms containing out-of-bounds indices are deleted.

  SharedDofPtr<Vec<Float, ncomp>> &lo_coeffs = lo_elem.m_dof_ptr;
  const int32 lo_order = lo_elem.get_order();
  const int32 hi_order = hi_elem.get_order();
  const int32 r = raise;

  // TODO is there ASSERT on the device?
  assert(raise == hi_order - lo_order);

  uint64 denom = 1u;
  for (int32 p = lo_order + 1; p <= hi_order; p++)
    denom *= p;

  int32 rchoose[r+1];
  {
    BinomialCoeff binomial_coeff;
    binomial_coeff.construct(r);
    rchoose[0] = binomial_coeff.get_val();
    for (int32 s = 1; s <= r; s++)
      rchoose[s] = binomial_coeff.slide_over(0);
  }

  // Factors depend on i and s.
  uint64 products0_L[r+1], products1_L[r+1], products2_L[r+1];
  uint64 products0_R[r+1], products1_R[r+1], products2_R[r+1];
  products0_L[r] = products1_L[r] = products2_L[r] = 1;
  products0_R[0] = products1_R[0] = products2_R[0] = 1;

  // TODO there should be an option to only write the interior dofs. (0 < I < hi_order)

  const int32 P0 = (dim >= 1 ? hi_order : 0);
  const int32 P1 = (dim >= 2 ? hi_order : 0);
  const int32 P2 = (dim >= 3 ? hi_order : 0);

  int32 out_idx = 0;
  for (int32 I2 = 0; I2 <= P2; I2++)
  {
    int32 min_s2 = (r - I2 >= 0 ? r - I2 : 0);
    int32 max_s2 = (hi_order - I2 <= r ? hi_order - I2 : r);

    for (int32 s2 = r-1; s2 >= min_s2; s2--)
      products2_L[s2] = products2_L[s2 + 1] * (I2 - (r-1 - s2));
    if (dim >= 3)
      for (int32 s2 = 1; s2 <= max_s2; s2++)
        products2_R[s2] = products2_R[s2 - 1] * (hi_order-I2 - (s2 - 1));
    else
      products2_R[r] = denom;

    for (int32 I1 = 0; I1 <= P1; I1++)
    {
      int32 min_s1 = (r - I1 >= 0 ? r - I1 : 0);
      int32 max_s1 = (hi_order - I1 <= r ? hi_order - I1 : r);

      for (int32 s1 = r-1; s1 >= min_s1; s1--)
        products1_L[s1] = products1_L[s1 + 1] * (I1 - (r-1 - s1));
      if (dim >= 2)
        for (int32 s1 = 1; s1 <= max_s1; s1++)
          products1_R[s1] = products1_R[s1 - 1] * (hi_order-I1 - (s1 - 1));
      else
        products1_R[r] = denom;

      for (int32 I0 = 0; I0 <= P0; I0++)
      {
        int32 min_s0 = (r - I0 >= 0 ? r - I0 : 0);
        int32 max_s0 = (hi_order - I0 <= r ? hi_order - I0 : r);

        // Assume dim >= 1
        //
        for (int32 s0 = r-1; s0 >= min_s0; s0--)
          products0_L[s0] = products0_L[s0 + 1] * (I0 - (r-1 - s0));
        for (int32 s0 = 1; s0 <= max_s0; s0++)
          products0_R[s0] = products0_R[s0 - 1] * (hi_order-I0 - (s0 - 1));

        hi_coeffs[out_idx] = 0;

        for (int32 s2 = min_s2; s2 <= max_s2; s2++)
        {
          const float64 premult2 = 1.0 * rchoose[s2] * products2_L[s2] * products2_R[s2] / denom;
          for (int32 s1 = min_s1; s1 <= max_s1; s1++)
          {
            const float64 premult1 = 1.0 * rchoose[s1] * products1_L[s1] * products1_R[s1] / denom * premult2;
            for (int32 s0 = min_s0; s0 <= max_s0; s0++)
            {
              const float64 premult0 = 1.0 * rchoose[s0] * products0_L[s0] * products0_R[s0] / denom * premult1;
              const int32 offset = (I2 - r + s2) * (lo_order+1) * (lo_order+1)
                                 + (I1 - r + s1) * (lo_order+1)
                                 + (I0 - r + s0);

              hi_coeffs[out_idx] += lo_coeffs[offset] * premult0;
            }
          }
        }

        out_idx++;
      }
    }
  }

}



// ---------------------------------------------------------------------------


// Template specialization (Tensor type, 0th order).
//
template <uint32 dim, uint32 ncomp>
class Element_impl<dim, ncomp, ElemType::Quad, Order::Constant> : public QuadRefSpace<dim>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 0;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs ()
  {
    return 1;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs (int32)
  {
    return 1;
  }

  // Get value without derivative.
  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, dim> &ref_coords) const
  {
    return *m_dof_ptr;
  }

  // Get value with derivative.
  DRAY_EXEC Vec<Float, ncomp> eval_d (const Vec<Float, dim> &ref_coords,
                                      Vec<Vec<Float, ncomp>, dim> &out_derivs) const
  {
    for (int d = 0; d < dim; d++)
      out_derivs[d] = 0;

    return *m_dof_ptr;
  }
};


// Template specialization (Quad type, 1st order, 2D).
//
template <uint32 ncomp>
class Element_impl<2u, ncomp, ElemType::Quad, Order::Linear> : public QuadRefSpace<2u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 1;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs ()
  {
    return 4;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs (int32)
  {
    return 4;
  }

  // Get value without derivative.
  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 2u> &r) const
  {
    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) + m_dof_ptr[1] * r[0] * (1 - r[1]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] + m_dof_ptr[3] * r[0] * r[1];
  }

  // Get value with derivative.
  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 2u> &r, Vec<Vec<Float, ncomp>, 2u> &out_derivs) const
  {
    out_derivs[0] = (m_dof_ptr[1] - m_dof_ptr[0]) * (1 - r[1]) +
                    (m_dof_ptr[3] - m_dof_ptr[2]) * r[1];

    out_derivs[1] = (m_dof_ptr[2] - m_dof_ptr[0]) * (1 - r[0]) +
                    (m_dof_ptr[3] - m_dof_ptr[1]) * r[0];

    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) + m_dof_ptr[1] * r[0] * (1 - r[1]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] + m_dof_ptr[3] * r[0] * r[1];
  }
};


// Template specialization (Quad type, 1st order, 3D).
//
template <uint32 ncomp>
class Element_impl<3u, ncomp, ElemType::Quad, Order::Linear> : public QuadRefSpace<3u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 1;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs ()
  {
    return 8;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs (int32)
  {
    return 8;
  }

  DRAY_EXEC void get_sub_bounds (const AABB<3> &sub_ref, AABB<ncomp> &aabb) const
  {
    using PtrT = SharedDofPtr<Vec<Float, ncomp>>;
    constexpr int32 POrder = 1;
    MultiVec<Float, 3u, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<3, ncomp, POrder, PtrT> (sub_ref.m_ranges, m_dof_ptr);
    for (int32 ii = 0; ii < 8; ii++)
       aabb.include (sub_nodes.linear_idx (ii));
  }

  // Get value without derivative.
  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 3u> &r) const
  {
    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[1] * r[0] * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] * (1 - r[2]) +
           m_dof_ptr[3] * r[0] * r[1] * (1 - r[2]) +
           m_dof_ptr[4] * (1 - r[0]) * (1 - r[1]) * r[2] +
           m_dof_ptr[5] * r[0] * (1 - r[1]) * r[2] +
           m_dof_ptr[6] * (1 - r[0]) * r[1] * r[2] + m_dof_ptr[7] * r[0] * r[1] * r[2];
  }

  // Get value with derivative.
  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 3u> &r, Vec<Vec<Float, ncomp>, 3u> &out_derivs) const
  {
    out_derivs[0] = (m_dof_ptr[1] - m_dof_ptr[0]) * (1 - r[1]) * (1 - r[2]) +
                    (m_dof_ptr[3] - m_dof_ptr[2]) * r[1] * (1 - r[2]) +
                    (m_dof_ptr[5] - m_dof_ptr[4]) * (1 - r[1]) * r[2] +
                    (m_dof_ptr[7] - m_dof_ptr[6]) * r[1] * r[2];

    out_derivs[1] = (m_dof_ptr[2] - m_dof_ptr[0]) * (1 - r[0]) * (1 - r[2]) +
                    (m_dof_ptr[3] - m_dof_ptr[1]) * r[0] * (1 - r[2]) +
                    (m_dof_ptr[6] - m_dof_ptr[4]) * (1 - r[0]) * r[2] +
                    (m_dof_ptr[7] - m_dof_ptr[5]) * r[0] * r[2];

    out_derivs[2] = (m_dof_ptr[4] - m_dof_ptr[0]) * (1 - r[0]) * (1 - r[1]) +
                    (m_dof_ptr[5] - m_dof_ptr[1]) * r[0] * (1 - r[1]) +
                    (m_dof_ptr[6] - m_dof_ptr[2]) * (1 - r[0]) * r[1] +
                    (m_dof_ptr[7] - m_dof_ptr[3]) * r[0] * r[1];

    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[1] * r[0] * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] * (1 - r[2]) +
           m_dof_ptr[3] * r[0] * r[1] * (1 - r[2]) +
           m_dof_ptr[4] * (1 - r[0]) * (1 - r[1]) * r[2] +
           m_dof_ptr[5] * r[0] * (1 - r[1]) * r[2] +
           m_dof_ptr[6] * (1 - r[0]) * r[1] * r[2] + m_dof_ptr[7] * r[0] * r[1] * r[2];
  }
};


// Template specialization (Quad type, 2nd order, 2D).
//
template <uint32 ncomp>
class Element_impl<2u, ncomp, ElemType::Quad, Order::Quadratic> : public QuadRefSpace<2u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 2;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs ()
  {
    return IntPow<3, 2u>::val;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs (int32)
  {
    return IntPow<3, 2u>::val;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 2u> &r) const
  {
    // Shape functions. Quadratic has 3 1D shape functions on each axis.
    Float su[3] = { (1 - r[0]) * (1 - r[0]), 2 * r[0] * (1 - r[0]), r[0] * r[0] };
    Float sv[3] = { (1 - r[1]) * (1 - r[1]), 2 * r[1] * (1 - r[1]), r[1] * r[1] };

    return m_dof_ptr[0] * su[0] * sv[0] + m_dof_ptr[1] * su[1] * sv[0] +
           m_dof_ptr[2] * su[2] * sv[0] + m_dof_ptr[3] * su[0] * sv[1] +
           m_dof_ptr[4] * su[1] * sv[1] + m_dof_ptr[5] * su[2] * sv[1] +
           m_dof_ptr[6] * su[0] * sv[2] + m_dof_ptr[7] * su[1] * sv[2] +
           m_dof_ptr[8] * su[2] * sv[2];
  }

  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 2u> &r, Vec<Vec<Float, ncomp>, 2u> &out_derivs) const
  {
    // Shape functions. Quadratic has 3 1D shape functions on each axis.
    Float su[3] = { (1 - r[0]) * (1 - r[0]), 2 * r[0] * (1 - r[0]), r[0] * r[0] };
    Float sv[3] = { (1 - r[1]) * (1 - r[1]), 2 * r[1] * (1 - r[1]), r[1] * r[1] };

    // Shape derivatives.
    Float dsu[3] = { -1 + r[0], 1 - r[0] - r[0], r[0] };
    Float dsv[3] = { -1 + r[1], 1 - r[1] - r[1], r[1] };

    out_derivs[0] = m_dof_ptr[0] * dsu[0] * sv[0] +
                    m_dof_ptr[1] * dsu[1] * sv[0] + m_dof_ptr[2] * dsu[2] * sv[0] +
                    m_dof_ptr[3] * dsu[0] * sv[1] + m_dof_ptr[4] * dsu[1] * sv[1] +
                    m_dof_ptr[5] * dsu[2] * sv[1] + m_dof_ptr[6] * dsu[0] * sv[2] +
                    m_dof_ptr[7] * dsu[1] * sv[2] + m_dof_ptr[8] * dsu[2] * sv[2];

    out_derivs[1] = m_dof_ptr[0] * su[0] * dsv[0] +
                    m_dof_ptr[1] * su[1] * dsv[0] + m_dof_ptr[2] * su[2] * dsv[0] +
                    m_dof_ptr[3] * su[0] * dsv[1] + m_dof_ptr[4] * su[1] * dsv[1] +
                    m_dof_ptr[5] * su[2] * dsv[1] + m_dof_ptr[6] * su[0] * dsv[2] +
                    m_dof_ptr[7] * su[1] * dsv[2] + m_dof_ptr[8] * su[2] * dsv[2];

    return m_dof_ptr[0] * su[0] * sv[0] + m_dof_ptr[1] * su[1] * sv[0] +
           m_dof_ptr[2] * su[2] * sv[0] + m_dof_ptr[3] * su[0] * sv[1] +
           m_dof_ptr[4] * su[1] * sv[1] + m_dof_ptr[5] * su[2] * sv[1] +
           m_dof_ptr[6] * su[0] * sv[2] + m_dof_ptr[7] * su[1] * sv[2] +
           m_dof_ptr[8] * su[2] * sv[2];
  }
};


// Template specialization (Quad type, 2nd order, 3D).
//
template <uint32 ncomp>
class Element_impl<3u, ncomp, ElemType::Quad, Order::Quadratic> : public QuadRefSpace<3u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 2;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs ()
  {
    return IntPow<3, 3u>::val;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs (int32)
  {
    return IntPow<3, 3u>::val;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 3u> &r) const
  {
    // TODO
    Vec<Float, ncomp> answer;
    answer = 0;
    return answer;
  }

  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 3u> &r, Vec<Vec<Float, ncomp>, 3u> &out_derivs) const
  {
    // Shape functions. Quadratic has 3 1D shape functions on each axis.
    Float su[3] = { (1 - r[0]) * (1 - r[0]), 2 * r[0] * (1 - r[0]), r[0] * r[0] };
    Float sv[3] = { (1 - r[1]) * (1 - r[1]), 2 * r[1] * (1 - r[1]), r[1] * r[1] };
    Float sw[3] = { (1 - r[2]) * (1 - r[2]), 2 * r[2] * (1 - r[2]), r[2] * r[2] };

    // Shape derivatives.
    Float dsu[3] = { -1 + r[0], 1 - r[0] - r[0], r[0] };
    Float dsv[3] = { -1 + r[1], 1 - r[1] - r[1], r[1] };
    Float dsw[3] = { -1 + r[2], 1 - r[2] - r[2], r[2] };

    out_derivs[0] =
    m_dof_ptr[0] * dsu[0] * sv[0] * sw[0] +
    m_dof_ptr[1] * dsu[1] * sv[0] * sw[0] + m_dof_ptr[2] * dsu[2] * sv[0] * sw[0] +
    m_dof_ptr[3] * dsu[0] * sv[1] * sw[0] + m_dof_ptr[4] * dsu[1] * sv[1] * sw[0] +
    m_dof_ptr[5] * dsu[2] * sv[1] * sw[0] + m_dof_ptr[6] * dsu[0] * sv[2] * sw[0] +
    m_dof_ptr[7] * dsu[1] * sv[2] * sw[0] + m_dof_ptr[8] * dsu[2] * sv[2] * sw[0] +

    m_dof_ptr[9] * dsu[0] * sv[0] * sw[1] + m_dof_ptr[10] * dsu[1] * sv[0] * sw[1] +
    m_dof_ptr[11] * dsu[2] * sv[0] * sw[1] +
    m_dof_ptr[12] * dsu[0] * sv[1] * sw[1] + m_dof_ptr[13] * dsu[1] * sv[1] * sw[1] +
    m_dof_ptr[14] * dsu[2] * sv[1] * sw[1] + m_dof_ptr[15] * dsu[0] * sv[2] * sw[1] +
    m_dof_ptr[16] * dsu[1] * sv[2] * sw[1] + m_dof_ptr[17] * dsu[2] * sv[2] * sw[1] +

    m_dof_ptr[18] * dsu[0] * sv[0] * sw[2] +
    m_dof_ptr[19] * dsu[1] * sv[0] * sw[2] + m_dof_ptr[20] * dsu[2] * sv[0] * sw[2] +
    m_dof_ptr[21] * dsu[0] * sv[1] * sw[2] + m_dof_ptr[22] * dsu[1] * sv[1] * sw[2] +
    m_dof_ptr[23] * dsu[2] * sv[1] * sw[2] + m_dof_ptr[24] * dsu[0] * sv[2] * sw[2] +
    m_dof_ptr[25] * dsu[1] * sv[2] * sw[2] + m_dof_ptr[26] * dsu[2] * sv[2] * sw[2];

    out_derivs[1] =
    m_dof_ptr[0] * su[0] * dsv[0] * sw[0] +
    m_dof_ptr[1] * su[1] * dsv[0] * sw[0] + m_dof_ptr[2] * su[2] * dsv[0] * sw[0] +
    m_dof_ptr[3] * su[0] * dsv[1] * sw[0] + m_dof_ptr[4] * su[1] * dsv[1] * sw[0] +
    m_dof_ptr[5] * su[2] * dsv[1] * sw[0] + m_dof_ptr[6] * su[0] * dsv[2] * sw[0] +
    m_dof_ptr[7] * su[1] * dsv[2] * sw[0] + m_dof_ptr[8] * su[2] * dsv[2] * sw[0] +

    m_dof_ptr[9] * su[0] * dsv[0] * sw[1] + m_dof_ptr[10] * su[1] * dsv[0] * sw[1] +
    m_dof_ptr[11] * su[2] * dsv[0] * sw[1] +
    m_dof_ptr[12] * su[0] * dsv[1] * sw[1] + m_dof_ptr[13] * su[1] * dsv[1] * sw[1] +
    m_dof_ptr[14] * su[2] * dsv[1] * sw[1] + m_dof_ptr[15] * su[0] * dsv[2] * sw[1] +
    m_dof_ptr[16] * su[1] * dsv[2] * sw[1] + m_dof_ptr[17] * su[2] * dsv[2] * sw[1] +

    m_dof_ptr[18] * su[0] * dsv[0] * sw[2] +
    m_dof_ptr[19] * su[1] * dsv[0] * sw[2] + m_dof_ptr[20] * su[2] * dsv[0] * sw[2] +
    m_dof_ptr[21] * su[0] * dsv[1] * sw[2] + m_dof_ptr[22] * su[1] * dsv[1] * sw[2] +
    m_dof_ptr[23] * su[2] * dsv[1] * sw[2] + m_dof_ptr[24] * su[0] * dsv[2] * sw[2] +
    m_dof_ptr[25] * su[1] * dsv[2] * sw[2] + m_dof_ptr[26] * su[2] * dsv[2] * sw[2];

    out_derivs[2] =
    m_dof_ptr[0] * su[0] * sv[0] * dsw[0] +
    m_dof_ptr[1] * su[1] * sv[0] * dsw[0] + m_dof_ptr[2] * su[2] * sv[0] * dsw[0] +
    m_dof_ptr[3] * su[0] * sv[1] * dsw[0] + m_dof_ptr[4] * su[1] * sv[1] * dsw[0] +
    m_dof_ptr[5] * su[2] * sv[1] * dsw[0] + m_dof_ptr[6] * su[0] * sv[2] * dsw[0] +
    m_dof_ptr[7] * su[1] * sv[2] * dsw[0] + m_dof_ptr[8] * su[2] * sv[2] * dsw[0] +

    m_dof_ptr[9] * su[0] * sv[0] * dsw[1] + m_dof_ptr[10] * su[1] * sv[0] * dsw[1] +
    m_dof_ptr[11] * su[2] * sv[0] * dsw[1] +
    m_dof_ptr[12] * su[0] * sv[1] * dsw[1] + m_dof_ptr[13] * su[1] * sv[1] * dsw[1] +
    m_dof_ptr[14] * su[2] * sv[1] * dsw[1] + m_dof_ptr[15] * su[0] * sv[2] * dsw[1] +
    m_dof_ptr[16] * su[1] * sv[2] * dsw[1] + m_dof_ptr[17] * su[2] * sv[2] * dsw[1] +

    m_dof_ptr[18] * su[0] * sv[0] * dsw[2] +
    m_dof_ptr[19] * su[1] * sv[0] * dsw[2] + m_dof_ptr[20] * su[2] * sv[0] * dsw[2] +
    m_dof_ptr[21] * su[0] * sv[1] * dsw[2] + m_dof_ptr[22] * su[1] * sv[1] * dsw[2] +
    m_dof_ptr[23] * su[2] * sv[1] * dsw[2] + m_dof_ptr[24] * su[0] * sv[2] * dsw[2] +
    m_dof_ptr[25] * su[1] * sv[2] * dsw[2] + m_dof_ptr[26] * su[2] * sv[2] * dsw[2];

    return m_dof_ptr[0] * su[0] * sv[0] * sw[0] +
           m_dof_ptr[1] * su[1] * sv[0] * sw[0] + m_dof_ptr[2] * su[2] * sv[0] * sw[0] +
           m_dof_ptr[3] * su[0] * sv[1] * sw[0] + m_dof_ptr[4] * su[1] * sv[1] * sw[0] +
           m_dof_ptr[5] * su[2] * sv[1] * sw[0] + m_dof_ptr[6] * su[0] * sv[2] * sw[0] +
           m_dof_ptr[7] * su[1] * sv[2] * sw[0] + m_dof_ptr[8] * su[2] * sv[2] * sw[0] +

           m_dof_ptr[9] * su[0] * sv[0] * sw[1] + m_dof_ptr[10] * su[1] * sv[0] * sw[1] +
           m_dof_ptr[11] * su[2] * sv[0] * sw[1] +
           m_dof_ptr[12] * su[0] * sv[1] * sw[1] +
           m_dof_ptr[13] * su[1] * sv[1] * sw[1] +
           m_dof_ptr[14] * su[2] * sv[1] * sw[1] +
           m_dof_ptr[15] * su[0] * sv[2] * sw[1] +
           m_dof_ptr[16] * su[1] * sv[2] * sw[1] +
           m_dof_ptr[17] * su[2] * sv[2] * sw[1] +

           m_dof_ptr[18] * su[0] * sv[0] * sw[2] +
           m_dof_ptr[19] * su[1] * sv[0] * sw[2] +
           m_dof_ptr[20] * su[2] * sv[0] * sw[2] +
           m_dof_ptr[21] * su[0] * sv[1] * sw[2] +
           m_dof_ptr[22] * su[1] * sv[1] * sw[2] +
           m_dof_ptr[23] * su[2] * sv[1] * sw[2] +
           m_dof_ptr[24] * su[0] * sv[2] * sw[2] +
           m_dof_ptr[25] * su[1] * sv[2] * sw[2] +
           m_dof_ptr[26] * su[2] * sv[2] * sw[2];
  }
};


// Template specialization (Quad type, 3rd order).
//
template <uint32 dim, uint32 ncomp>
class Element_impl<dim, ncomp, ElemType::Quad, Order::Cubic> : public QuadRefSpace<dim>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 3;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs ()
  {
    return IntPow<4, dim>::val;
  }
  DRAY_EXEC static constexpr int32 get_num_dofs (int32)
  {
    return IntPow<4, dim>::val;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, dim> &r) const
  {
    // TODO
    Vec<Float, ncomp> answer;
    answer = 0;
    return answer;
  }

  DRAY_EXEC Vec<Float, ncomp> eval_d (const Vec<Float, dim> &ref_coords,
                                      Vec<Vec<Float, ncomp>, dim> &out_derivs) const
  {
    // TODO
    Vec<Float, ncomp> answer;
    answer = 0;
    return answer;
  }
};


} // namespace dray

#endif // DRAY_POS_TENSOR_ELEMENT_HPP

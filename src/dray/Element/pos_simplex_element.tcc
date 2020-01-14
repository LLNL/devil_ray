// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POS_SIMPLEX_ELEMENT_HPP
#define DRAY_POS_SIMPLEX_ELEMENT_HPP

/**
 * @file pos_simplex_element.hpp
 * @brief Partial template specialization of Element_impl
 *        for simplex (i.e. tet and tri) elements.
 */

#include <dray/Element/element.hpp>
#include <dray/exports.hpp>
#include <dray/integer_utils.hpp> // MultinomialCoeff
#include <dray/vec.hpp>

namespace dray
{
//
// TriElement_impl
//
template <uint32 dim, uint32 ncomp, int32 P>
using TriElement_impl = Element_impl<dim, ncomp, ElemType::Tri, P>;


template <uint32 dim> class TriRefSpace
{
  public:
  DRAY_EXEC static bool is_inside (const Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static void clamp_to_domain (Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static Vec<Float, dim>
  project_to_domain (const Vec<Float, dim> &r1, const Vec<Float, dim> &r2); // TODO
};

template <uint32 dim> struct RefTri
{
  Vec<float32, dim> m_vertices[dim + 1];

  DRAY_EXEC static RefTri ref_universe ()
  {
    RefTri ret;
    for (int d = 0; d < dim; d++)
    {
      ret.m_vertices[d] = 0.0f;
      ret.m_vertices[d][d] = 1.0f;
      ret.m_vertices[dim][d] = 1.0f;
    }
    return ret;
  }

  DRAY_EXEC Vec<float32, dim> center () const
  {
    Vec<float32, dim> c;
    c = 0.0;
    for (int d = 0; d <= dim; d++)
      c += m_vertices[d];
    c *= float32 (1.0 / (dim + 1));
    return c;
  }

  DRAY_EXEC float32 max_length () const
  {
    // Any proxy for diameter. In this case use maximum edge length.
    float32 M = 0.0;
    for (int32 v1 = 0; v1 <= dim; v1++)
      for (int32 v2 = 0; v2 <= dim; v2++)
        M = fmaxf (M, (m_vertices[v1] - m_vertices[v2]).magnitude2 ());
    return sqrtf (M);
  }
};

// Specialize SubRef for Tri type.
template <> struct ElemTypeAttributes<ElemType::Tri>
{
  template <uint32 dim> using SubRef = RefTri<dim>;
};


/*
 * Example evaluation using Horner's rule.
 *
 * Dofs for a quadratic triangular element form a triangle.
 *
 * Barycentric coordinates u + v + t = 1  (u,v,t >= 0).
 *
 * To evaluate an element using Horner's rule, while accessing dofs
 * in the lexicographic order of (u fastest, v next fastest),
 * express the summation as follows:
 *
 *   v^2 * {  Mck(2;0,2,0)*u^0 }
 *   v^1 * { [Mck(2;0,1,1)*u^0]*t + Mck(2;1,1,0)*u^1 }
 *   v^0 * { [[Mck(2;0,0,2)*u^0]*t + Mck(2;1,0,1)*u^1]*t + Mck(2;2,0,0)*u^2 }
 *
 *           7  []   \`
 *          /         \`
 *    'v'  /           \` (t=0)
 *        /  []    []   \`
 *       /               \`
 *      /                 \`
 *        []    []    []
 *  (t=1) ------------->
 *             'u'
 *
 *  where 'Mck' stands for M choose K, or multinomial coefficient.
 *
 *  Note that multinomial coefficients are symmetric in the indices.
 *  In particular,
 *      Mck(2; 0,0,2) == Mck(2; 2,0,0)
 *      Mck(2; 0,1,1) == Mck(2; 1,1,0)
 *  This property allows us to traverse Pascal's simplex using only
 *  transitions between 'adjacent' multinomial coefficients.
 */


// ---------------------------------------------------------------------------

// Template specialization (Tri type, general order, 2D).
//
template <uint32 ncomp>
class Element_impl<2u, ncomp, ElemType::Tri, Order::General> : public TriRefSpace<2u>
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
    return (order + 1) * (order + 2) / 2;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 2u> &ref_coords) const;

  DRAY_EXEC Vec<Float, ncomp> eval_d (const Vec<Float, 2u> &ref_coords,
                                      Vec<Vec<Float, ncomp>, 2u> &out_derivs) const;

  DRAY_EXEC void get_sub_bounds (const RefTri<2u> &sub_ref, AABB<ncomp> &aabb) const;

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


// Template specialization (Tri type, general order, 3D).
//
template <uint32 ncomp>
class Element_impl<3u, ncomp, ElemType::Tri, Order::General> : public TriRefSpace<3u>
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
    return (order + 1) * (order + 2) * (order + 3) / 6;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 3u> &ref_coords) const;

  DRAY_EXEC Vec<Float, ncomp> eval_d (const Vec<Float, 3u> &ref_coords,
                                      Vec<Vec<Float, ncomp>, 3u> &out_derivs) const;

  DRAY_EXEC void get_sub_bounds (const RefTri<3u> &sub_ref, AABB<ncomp> &aabb) const;

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


// -----
// Implementations
// -----

template <uint32 dim>
DRAY_EXEC bool TriRefSpace<dim>::is_inside (const Vec<Float, dim> &ref_coords)
{
  Float min_val = 2.f;
  Float t = 1.0f;
  for (int32 d = 0; d < dim; d++)
  {
    min_val = min (ref_coords[d], min_val);
    t -= ref_coords[d];
  }
  min_val = min (t, min_val);
  return (min_val >= 0.f - epsilon<Float> ());
}

template <uint32 dim>
DRAY_EXEC void TriRefSpace<dim>::clamp_to_domain (Vec<Float, dim> &ref_coords)
{
  // TODO
}

template <uint32 dim>
DRAY_EXEC Vec<Float, dim>
TriRefSpace<dim>::project_to_domain (const Vec<Float, dim> &r1, const Vec<Float, dim> &r2)
{
  return { 0.0 }; // TODO
}

// ------------


// :: 2D :: //

//
// eval() (2D triangle evaluation)
//
template <uint32 ncomp>
DRAY_EXEC Vec<Float, ncomp>
Element_impl<2u, ncomp, ElemType::Tri, Order::General>::eval (const Vec<Float, 2u> &ref_coords) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  const uint32 p = m_order;
  PtrT dof_ptr = m_dof_ptr; // Make a local copy that can be incremented.

  // Barycentric coordinates.
  const Float &u = ref_coords[0];
  const Float &v = ref_coords[1];
  const Float t = Float (1.0) - (u + v);

  // Multinomial coefficient. Will traverse Pascal's simplex using
  // transitions between adjacent multinomial coefficients (slide_over()),
  // and transpositions back to the start of each row (swap_places()).
  MultinomialCoeff<2> mck;
  mck.construct (p);

  DofT j_sum;
  j_sum = 0.0;
  Float vpow = 1.0;
  for (int32 jj = 0; jj <= p; jj++)
  {

    DofT i_sum;
    i_sum = 0.0;
    Float upow = 1.0;
    for (int32 ii = 0; ii <= (p - jj); ii++)
    {
      // Horner's rule innermost, due to decreasing powers of t (mu = p - jj - ii).
      i_sum *= t;
      i_sum = i_sum + (*(dof_ptr)) * (mck.get_val () * upow);
      ++dof_ptr;
      upow *= u;
      if (ii < (p - jj)) mck.slide_over (0);
    }
    mck.swap_places (0);

    j_sum = j_sum + i_sum * vpow;
    vpow *= v;
    if (jj < p) mck.slide_over (1);
  }
  // mck.swap_places(1);

  return j_sum;
}


//
// eval_d() (2D triangle eval & derivatives)
//
template <uint32 ncomp>
DRAY_EXEC Vec<Float, ncomp> Element_impl<2u, ncomp, ElemType::Tri, Order::General>::eval_d (
const Vec<Float, 2u> &ref_coords,
Vec<Vec<Float, ncomp>, 2u> &out_derivs) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  if (m_order == 0)
  {
    out_derivs[0] = 0.0;
    out_derivs[1] = 0.0;
    return m_dof_ptr[0];
  }

  // The Bernstein--Bezier simplex basis has the following properties:
  //
  // - Derivatives in terms of (p-1)-order triangle:
  //     du = \sum_{i + j + \mu = p-1} \beta^{p-1}_{i, j, \mu} \left( C_{i+1, j, \mu} - C_{i, j, \mu+1} \right)
  //     dv = \sum_{i + j + \mu = p-1} \beta^{p-1}_{i, j, \mu} \left( C_{i, j+1, \mu} - C_{i, j, \mu+1} \right)
  //
  // - p-order triangle in terms of (p-1)-order triangle:
  //     F = \sum_{i + j + \mu = p-1} \beta^{p-1}_{i, j, \mu} \left( C_{i+1, j, \mu} u + C_{i, j+1, \mu} v + C_{i, j, \mu+1} t \right)

  // The dof offset in an axis depends on the index in that axis and lesser axes.
  // The offset can be derived from the linearization formula.
  // Note: D^d(p) is the number of dofs in a d-dimensional p-order simplex, or nchoosek(p+d,d).
  //
  //     l(i,j) = \sum_{j'=0}^{j-1} D^1(p-j') + \sum_{i'=0}^{i-1} D^0(p-j-i')
  //
  //            = ...
  //
  //            = D^2(p) - D^2(p-j) + D^1(p-j) - D^1(p-j-i)
  //
  //     \delta l^0 (i,j) = D^1(p-j-i) - D^1(p-j-i-1)
  //                      = D^0(p-j-i)
  //                      = 1
  //
  //     \delta l^1 (i,j) = D^1(p-j) - D^0(p-j) + D^0(p-j-i)
  //                      = D^1(p-j) - D^0(p-j) + \delta l^0 (i,j)
  //                      = p-j+1

  const uint32 p = m_order;
  const uint32 pm1 = m_order - 1;
  PtrT dof_ptr = m_dof_ptr; // Make a local copy that can be incremented.

  // Barycentric coordinates.
  const Float &u = ref_coords[0];
  const Float &v = ref_coords[1];
  const Float t = Float (1.0) - (u + v);

  // Multinomial coefficient. Will traverse Pascal's simplex using
  // transitions between adjacent multinomial coefficients (slide_over()),
  // and transpositions back to the start of each row (swap_places()).
  MultinomialCoeff<2> mck;
  mck.construct (pm1);

  int32 dof_idx = 0;

  DofT j_sum;
  j_sum = 0.0;
  Vec<DofT, 2u> j_sum_d;
  j_sum_d = 0.0;
  Float vpow = 1.0;
  for (int32 jj = 0; jj <= pm1; jj++)
  {
    const int32 sz_p_j = (p - jj + 1) / 1; // nchoosek(p-jj + dim-1, dim-1)

    DofT i_sum;
    i_sum = 0.0;
    Vec<DofT, 2u> i_sum_d;
    i_sum_d = 0.0;
    Float upow = 1.0;
    for (int32 ii = 0; ii <= (pm1 - jj); ii++)
    {
      // Horner's rule innermost, due to decreasing powers of t (mu = pm1 - jj - ii).
      i_sum *= t;
      i_sum_d[0] *= t;
      i_sum_d[1] *= t;

      const DofT dof_mu = dof_ptr[dof_idx];
      const Vec<DofT, 2u> dof_ij = { dof_ptr[dof_idx + 1], // Offset dofs
                                     dof_ptr[dof_idx + sz_p_j] };
      dof_idx++;

      i_sum += (dof_mu * t + dof_ij[0] * u + dof_ij[1] * v) * (mck.get_val () * upow);
      i_sum_d[0] += (dof_ij[0] - dof_mu) * (mck.get_val () * upow);
      i_sum_d[1] += (dof_ij[1] - dof_mu) * (mck.get_val () * upow);

      upow *= u;
      if (ii < (pm1 - jj)) mck.slide_over (0);
    }
    mck.swap_places (0);

    dof_idx++; // Skip end of row.

    j_sum += i_sum * vpow;
    j_sum_d += i_sum_d * vpow;
    vpow *= v;
    if (jj < pm1) mck.slide_over (1);
  }
  // mck.swap_places(1);

  out_derivs = j_sum_d * p;
  return j_sum;
}


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


template <uint32 ncomp>
template <uint32 raise>
DRAY_EXEC void Element_impl<2u, ncomp, ElemType::Tri, Order::General>
::project_to_higher_order_basis(const Element_impl &lo_elem,
                                Element_impl &hi_elem,
                                WriteDofPtr<Vec<Float, ncomp>> &hi_coeffs)
{
  SharedDofPtr<Vec<Float, ncomp>> &lo_coeffs = lo_elem.m_dof_ptr;
  const int32 lo_order = lo_elem.get_order();
  const int32 hi_order = hi_elem.get_order();
  const int32 r = raise;

  // TODO is there ASSERT on the device?
  assert(raise == hi_order - lo_order);

  // Copy the lo_elem coefficients to the hi_elem coefficients.
  int32 lo_idx = 0;
  int32 offset_ii = 0;
  for (int32 jj = 0; jj <= lo_order; jj++)
  {
    for (int32 ii = 0; ii <= lo_order - jj; ii++)
      hi_coeffs[offset_ii + ii] = lo_coeffs[lo_idx++];
    offset_ii += (hi_order + 1 - jj);  // Stack a 1D triangle.
  }

  // Apply DeCasteljau degree raising, by one order at a time.
  for (uint32 p = lo_order; p < hi_order; p++)
  {
    // From order (p) to order (p+1).

    using detail::cartesian_to_tri_idx;

    // Initialize outer edge to 0.
    for (int32 ii = 0; ii <= p+1; ii++)
    {
      const int32 jj = p+1 - ii;
      const int32 hi_idx = cartesian_to_tri_idx(ii, jj, hi_order+1);
      hi_coeffs[hi_idx] = 0;
    }

    const Float rcp_p1 = 1.0 / (p+1);

    // Spread the coefficients toward the outward edge,
    // working our way inwards so we can do it in place.
    for (int32 mu = 0; mu <= p; mu++)
    {
      for (int32 ii = 0; ii <= p-mu; ii++)
      {
        const int32 jj = p - mu - ii;
        const int32 read_idx = cartesian_to_tri_idx(ii, jj, hi_order+1);
        hi_coeffs[cartesian_to_tri_idx(ii+1, jj, hi_order+1)]
            += hi_coeffs[read_idx] * ((ii+1) * rcp_p1);
        hi_coeffs[cartesian_to_tri_idx(ii, jj+1, hi_order+1)]
            += hi_coeffs[read_idx] * ((jj+1) * rcp_p1);
        hi_coeffs[cartesian_to_tri_idx(ii, jj, hi_order+1)]
            = hi_coeffs[read_idx] * ((mu+1) * rcp_p1);
      }
    }
  }
}


template <uint32 ncomp>
DRAY_EXEC void
Element_impl<2u, ncomp, ElemType::Tri, Order::General>::get_sub_bounds (const RefTri<2u> &sub_ref,
                                                                        AABB<ncomp> &aabb) const
{
  // Take an arbitrary sub-triangle in reference space, and return bounds
  // on the function restricted to that sub-triangle.

  // TODO TODO
  //
  // Use the results of
  //
  // @article{derose1988composing,
  //   title={Composing b{\'e}zier simplexes},
  //   author={DeRose, Tony D},
  //   journal={ACM Transactions on Graphics (TOG)},
  //   volume={7},
  //   number={3},
  //   pages={198--221},
  //   year={1988},
  //   publisher={ACM}
  //  }

  // As a PLACEHOLDER STUB ONLY, return bounds on the entire element.
  // NOTE: This will defeat subdivision searches. It will cause the search space
  // to increase rather than decrease on each step.

  aabb.reset ();
  const int num_dofs = get_num_dofs ();
  for (int ii = 0; ii < num_dofs; ii++)
    aabb.include (m_dof_ptr[ii]);
}


// :: 3D :: //

//
// eval() (3D tetrahedron evaluation)
//
template <uint32 ncomp>
DRAY_EXEC Vec<Float, ncomp>
Element_impl<3u, ncomp, ElemType::Tri, Order::General>::eval (const Vec<Float, 3u> &ref_coords) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  const unsigned int p = m_order;
  PtrT dof_ptr = m_dof_ptr; // Make a local copy that can be incremented.

  // Barycentric coordinates.
  const Float &u = ref_coords[0];
  const Float &v = ref_coords[1];
  const Float &w = ref_coords[2];
  const Float t = Float (1.0) - (u + v + w);

  // Multinomial coefficient. Will traverse Pascal's simplex using
  // transitions between adjacent multinomial coefficients (slide_over()),
  // and transpositions back to the start of each row (swap_places()).
  MultinomialCoeff<3> mck;
  mck.construct (p);

  DofT k_sum;
  k_sum = 0.0;
  Float wpow = 1.0;
  for (int32 kk = 0; kk <= p; kk++)
  {

    DofT j_sum;
    j_sum = 0.0;
    Float vpow = 1.0;
    for (int32 jj = 0; jj <= p - kk; jj++)
    {

      DofT i_sum;
      i_sum = 0.0;
      Float upow = 1.0;
      for (int32 ii = 0; ii <= (p - kk - jj); ii++)
      {
        // Horner's rule innermost, due to decreasing powers of t (mu = p - kk - jj - ii).
        i_sum *= t;
        i_sum += (*dof_ptr) * (mck.get_val () * upow);
        ++dof_ptr;
        upow *= u;
        if (ii < (p - kk - jj)) mck.slide_over (0);
      }
      mck.swap_places (0);

      j_sum += i_sum * vpow;
      vpow *= v;
      if (jj < p - kk) mck.slide_over (1);
    }
    mck.swap_places (1);

    k_sum += j_sum * wpow;
    wpow *= w;
    if (kk < p) mck.slide_over (2);
  }
  // mck.swap_places(2);

  return k_sum;
}


//
// eval_d() (3D tetrahedron eval & derivatives)
//
template <uint32 ncomp>
DRAY_EXEC Vec<Float, ncomp> Element_impl<3u, ncomp, ElemType::Tri, Order::General>::eval_d (
const Vec<Float, 3u> &ref_coords,
Vec<Vec<Float, ncomp>, 3u> &out_derivs) const
{
  using DofT = Vec<Float, ncomp>;
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  if (m_order == 0)
  {
    out_derivs[0] = 0.0;
    out_derivs[1] = 0.0;
    out_derivs[2] = 0.0;
    return m_dof_ptr[0];
  }

  // The dof offset in an axis depends on the index in that axis and lesser axes.
  // The offset can be derived from the linearization formula.
  // Note: D^d(p) is the number of dofs in a d-dimensional p-order simplex, or nchoosek(p+d,d).
  //
  //     l(i,j,k) =   \sum_{k'=0}^[k-1} D^2(p-k')
  //                + \sum_{j'=0}^{j-1} D^1(p-k-j')
  //                + \sum_{i'=0}^{i-1} D^0(p-k-j-i')
  //
  //              = ...
  //
  //              =   D^3(p)     - D^3(p-k)
  //                + D^2(p-k)   - D^2(p-k-j)
  //                + D^1(p-k-j) - D^1(p-k-j-i)
  //
  //     \delta l^0 (i,j,k) = D^0(p-k-j-i)  = 1
  //
  //     \delta l^1 (i,j,k) = D^1(p-k-j) - D^0(p-k-j) + D^0(p-k-j-i)
  //                        = D^1(p-k-j) - D^0(p-k-j) + \delta l^0 (i,j,k)  = p-k-j+1
  //
  //     \delta l^2 (i,j,k) = D^2(p-k) - D^1(p-k) + D^1(p-k-j) - D^0(p-k-j) + D^0(p-k-j-i)
  //                        = D^2(p-k) - D^1(p-k) + \delta l^1(i,j,k)   = (p-k+1)(p-k+2)/2 - j

  const uint32 p = m_order;
  const uint32 pm1 = m_order - 1;
  PtrT dof_ptr = m_dof_ptr; // Make a local copy that can be incremented.

  // Barycentric coordinates.
  const Float &u = ref_coords[0];
  const Float &v = ref_coords[1];
  const Float &w = ref_coords[2];
  const Float t = Float (1.0) - (u + v + w);

  // Multinomial coefficient. Will traverse Pascal's simplex using
  // transitions between adjacent multinomial coefficients (slide_over()),
  // and transpositions back to the start of each row (swap_places()).
  MultinomialCoeff<3> mck;
  mck.construct (pm1);

  int32 dof_idx = 0;

  DofT k_sum;
  k_sum = 0.0;
  Vec<DofT, 3u> k_sum_d;
  k_sum_d = 0.0;
  Float wpow = 1.0;
  for (int32 kk = 0; kk <= pm1; kk++)
  {
    const int32 sz_p_k = (p - kk + 1) * (p - kk + 2) / (1 * 2); // nchoosek(p-kk + dim-1, dim-1)

    DofT j_sum;
    j_sum = 0.0;
    Vec<DofT, 3u> j_sum_d;
    j_sum_d = 0.0;
    Float vpow = 1.0;
    for (int32 jj = 0; jj <= (pm1 - kk); jj++)
    {
      const int32 sz_p_j = (p - kk - jj + 1) / 1; // nchoosek(q-jj + dim-2, dim-2)

      DofT i_sum;
      i_sum = 0.0;
      Vec<DofT, 3u> i_sum_d;
      i_sum_d = 0.0;
      Float upow = 1.0;
      for (int32 ii = 0; ii <= (pm1 - kk - jj); ii++)
      {
        // Horner's rule innermost, due to decreasing powers of t (mu = pm1 - kk - jj - ii).
        i_sum *= t;
        i_sum_d[0] *= t;
        i_sum_d[1] *= t;
        i_sum_d[2] *= t;

        const DofT dof_mu = dof_ptr[dof_idx];
        const Vec<DofT, 3u> dof_ijk = { dof_ptr[dof_idx + 1], // Offset dofs
                                        dof_ptr[dof_idx + sz_p_j],
                                        dof_ptr[dof_idx + sz_p_k - jj] };
        dof_idx++;

        i_sum += (dof_mu * t + dof_ijk[0] * u + dof_ijk[1] * v + dof_ijk[2] * w) *
                 (mck.get_val () * upow);
        i_sum_d[0] += (dof_ijk[0] - dof_mu) * (mck.get_val () * upow);
        i_sum_d[1] += (dof_ijk[1] - dof_mu) * (mck.get_val () * upow);
        i_sum_d[2] += (dof_ijk[2] - dof_mu) * (mck.get_val () * upow);

        upow *= u;
        if (ii < (pm1 - kk - jj)) mck.slide_over (0);
      }
      mck.swap_places (0);

      dof_idx++; // Skip end of row.

      j_sum += i_sum * vpow;
      j_sum_d += i_sum_d * vpow;
      vpow *= v;
      if (jj < (pm1 - kk)) mck.slide_over (1);
    }
    mck.swap_places (1);

    dof_idx++; // Skip tip of triangle.

    k_sum += j_sum * wpow;
    k_sum_d += j_sum_d * wpow;
    wpow *= w;
    if (kk < pm1) mck.slide_over (2);
  }
  // mck.swap_places(2);

  out_derivs = k_sum_d * p;
  return k_sum;
}


template <uint32 ncomp>
template <uint32 raise>
DRAY_EXEC void
Element_impl<3u, ncomp, ElemType::Tri, Order::General>
::project_to_higher_order_basis(const Element_impl &lo_elem,
                                Element_impl &hi_elem,
                                WriteDofPtr<Vec<Float, ncomp>> &hi_coeffs)
{
  SharedDofPtr<Vec<Float, ncomp>> &lo_coeffs = lo_elem.m_dof_ptr;
  const int32 lo_order = lo_elem.get_order();
  const int32 hi_order = hi_elem.get_order();
  const int32 r = raise;

  // TODO is there ASSERT on the device?
  assert(raise == hi_order - lo_order);

  // Copy the lo_elem coefficients to the hi_elem coefficients.
  int32 lo_idx = 0;
  int32 offset_jj = 0;
  for (int32 kk = 0; kk <= lo_order; kk++)
  {
    int32 offset_ii = offset_jj;
    for (int32 jj = 0; jj <= lo_order; jj++)
    {
      for (int32 ii = 0; ii <= lo_order - jj; ii++)
        hi_coeffs[offset_ii + ii] = lo_coeffs[lo_idx++];
      offset_ii += (hi_order + 1 - jj);  // Add 1D triangle.
    }
    offset_jj += (hi_order + 1 - kk) * (hi_order + 2 - kk) / 2;  // Add 2D triangle.
  }

  // Apply DeCasteljau degree raising, by one order at a time.
  for (uint32 p = lo_order; p < hi_order; p++)
  {
    // From order (p) to order (p+1).

    using detail::cartesian_to_tet_idx;

    // Initialize outer face to 0.
    for (int32 jj = 0; jj <= p+1; jj++)
      for (int32 ii = 0; ii <= p+1 - jj; ii++)
      {
        const int32 kk = p+1 -jj - ii;
        const int32 hi_idx = cartesian_to_tet_idx(ii, jj, kk, hi_order+1);
        hi_coeffs[hi_idx] = 0;
      }

    const Float rcp_p1 = 1.0 / (p+1);

    // Spread the coefficients toward the outward face,
    // working our way inwards so we can do it in place.
    for (int32 mu = 0; mu <= p; mu++)
    {
      for (int32 jj = 0; jj <= p-mu; jj++)
        for (int32 ii = 0; ii <= p-mu - jj; ii++)
        {
          const int32 kk = p - mu - jj - ii;
          const int32 read_idx = cartesian_to_tet_idx(ii, jj, kk, hi_order+1);
          hi_coeffs[cartesian_to_tet_idx(ii+1, jj, kk, hi_order+1)]
              += hi_coeffs[read_idx] * ((ii+1) * rcp_p1);
          hi_coeffs[cartesian_to_tet_idx(ii, jj+1, kk, hi_order+1)]
              += hi_coeffs[read_idx] * ((jj+1) * rcp_p1);
          hi_coeffs[cartesian_to_tet_idx(ii, jj, kk+1, hi_order+1)]
              += hi_coeffs[read_idx] * ((kk+1) * rcp_p1);
          hi_coeffs[cartesian_to_tet_idx(ii, jj, kk, hi_order+1)]
              = hi_coeffs[read_idx] * ((mu+1) * rcp_p1);
        }
    }
  }
}


template <uint32 ncomp>
DRAY_EXEC void
Element_impl<3u, ncomp, ElemType::Tri, Order::General>::get_sub_bounds (const RefTri<3u> &sub_ref,
                                                                        AABB<ncomp> &aabb) const
{
  // Take an arbitrary sub-tetrahedron in reference space, and return bounds
  // on the function restricted to that sub-tetrahedron.

  // TODO TODO
  //
  // Use the results of
  //
  // @article{derose1988composing,
  //   title={Composing b{\'e}zier simplexes},
  //   author={DeRose, Tony D},
  //   journal={ACM Transactions on Graphics (TOG)},
  //   volume={7},
  //   number={3},
  //   pages={198--221},
  //   year={1988},
  //   publisher={ACM}
  //  }

  // As a PLACEHOLDER STUB ONLY, return bounds on the entire element.
  // NOTE: This will defeat subdivision searches. It will cause the search space
  // to increase rather than decrease on each step.

  aabb.reset ();
  const int num_dofs = get_num_dofs ();
  for (int ii = 0; ii < num_dofs; ii++)
    aabb.include (m_dof_ptr[ii]);
}


/** @brief Specialization of SplitRefBox from subdivision_search.hpp for triangles. */
namespace detail
{
template <uint32 dim> struct SplitRefBox<RefTri<dim>>
{
  DRAY_EXEC static void
  split_ref_box (int32 depth, const RefTri<dim> &p, RefTri<dim> &child1, RefTri<dim> &child2)
  {
#warning "SplitRefBox<RefTri>::split_ref_box() demands a redesign!"

    // There is a way to subdivide a tetrahedron into 8 smaller tetrahedra,
    // if we are willing for the subdivision to be asymmetrical.
    // Dividing a triangle into 4 smaller triangles is symmetrical.

    // With the existing interface we need binary splits. Employ depth to accomplish this.
    //  Triangle:       (depth)          Tetrahedron:
    // -------------                -------------------------------------------
    //     [P]             0           [P]
    //    /   \                       /   \
    //  [0]    +           1        [0]    +        Children 0--3 are
    //        / \                         / \            tips of parent.
    //     [1]   +         2           [1]   +
    //          / \                         / \
    //       [2]  [3]      3             [2]   +        Splitting the central octahedron:
    //                                        / \
    //                     4               [3]   +   [4] (top) {c/2, (a+b)/4, (a+c)/2, (b+c)/2}
    //                                          / \  [5] (front) {(a+b)/4, (a+c)/2, (b+c)/2, (a+b)/2}
    //                     5                 [4]   +   [6] (left) {(a+b)/4, (a+c)/2, a/2, (a+b)/2}
    //                                            / \  [7] (right) {(a+b)/4, (b+c)/2, b/2, (a+b)/2}
    //                     6                   [5]   +
    //                                              / \
    //                     7                     [6]   [7]

    const int32 periodicity = (dim == 2 ? 3 : dim == 3 ? 7 : 1);
    const int32 chnum = depth % periodicity;
    const bool right_is_child = (dim == 2 && chnum == 2) || (dim == 3 && chnum == 6);

    if (dim == 2)
    {
      // Left child.
      if (chnum == 0)
      {
        child1.m_vertices[0] = (p.m_vertices[chnum] + p.m_vertices[1]) * 0.5f;
        child1.m_vertices[1] = (p.m_vertices[chnum] + p.m_vertices[2]) * 0.5f;
        child1.m_vertices[dim] = p.m_vertices[chnum];
      }
      if (chnum == 1)
      {
        child1.m_vertices[0] = (p.m_vertices[chnum] + p.m_vertices[0]) * 0.5f;
        child1.m_vertices[1] = (p.m_vertices[chnum] + p.m_vertices[2]) * 0.5f;
        child1.m_vertices[dim] = p.m_vertices[chnum];
      }
      if (chnum == 2)
      {
        child1.m_vertices[0] = (p.m_vertices[chnum] + p.m_vertices[0]) * 0.5f;
        child1.m_vertices[1] = (p.m_vertices[chnum] + p.m_vertices[1]) * 0.5f;
        child1.m_vertices[dim] = p.m_vertices[chnum];
      }

      // Right child or self.
      if (right_is_child)
      {
        child2.m_vertices[0] = (p.m_vertices[0] + p.m_vertices[1]) * 0.5f;
        child2.m_vertices[1] = (p.m_vertices[1] + p.m_vertices[2]) * 0.5f;
        child2.m_vertices[2] = (p.m_vertices[2] + p.m_vertices[0]) * 0.5f;
      }
      else
        child2 = p;
    }

    else if (dim == 3)
    {
      Vec<float32, dim> o = p.m_vertices[0];
      Vec<float32, dim> a = p.m_vertices[1];
      Vec<float32, dim> b = p.m_vertices[2];
      Vec<float32, dim> c = p.m_vertices[3];

      // Left child.
      switch (chnum)
      {
      case 0:
        child1.m_vertices[0] = (o + a) * 0.5f;
        child1.m_vertices[1] = (o + b) * 0.5f;
        child1.m_vertices[2] = (o + c) * 0.5f;
        child1.m_vertices[dim] = o;
        break;

      case 1:
        child1.m_vertices[0] = (a + o) * 0.5f;
        child1.m_vertices[1] = (a + b) * 0.5f;
        child1.m_vertices[2] = (a + c) * 0.5f;
        child1.m_vertices[dim] = a;
        break;

      case 2:
        child1.m_vertices[0] = (b + a) * 0.5f;
        child1.m_vertices[1] = (b + o) * 0.5f;
        child1.m_vertices[2] = (b + c) * 0.5f;
        child1.m_vertices[dim] = b;
        break;

      case 3:
        child1.m_vertices[0] = (c + a) * 0.5f;
        child1.m_vertices[1] = (c + b) * 0.5f;
        child1.m_vertices[2] = (c + o) * 0.5f;
        child1.m_vertices[dim] = c;
        break;


      case 4:
        child1.m_vertices[0] = (c + o) * 0.5f;
        child1.m_vertices[1] = (a + b) * 0.25f;
        child1.m_vertices[2] = (a + c) * 0.5f;
        child1.m_vertices[3] = (b + c) * 0.5f;
        break;

      case 5:
        child1.m_vertices[0] = (a + b) * 0.25f;
        child1.m_vertices[1] = (a + c) * 0.5f;
        child1.m_vertices[2] = (b + c) * 0.5f;
        child1.m_vertices[3] = (a + b) * 0.5f;
        break;

      case 6:
        child1.m_vertices[0] = (a + b) * 0.25f;
        child1.m_vertices[1] = (a + c) * 0.5f;
        child1.m_vertices[2] = (a + o) * 0.5f;
        child1.m_vertices[3] = (a + b) * 0.5f;
        break;
      }

      // Right child or self.
      if (right_is_child)
      {
        child2.m_vertices[0] = (a + b) * 0.25f;
        child2.m_vertices[1] = (b + c) * 0.5f;
        child2.m_vertices[2] = (b + o) * 0.5f;
        child2.m_vertices[3] = (a + b) * 0.5f;
      }
      else
        child2 = p;
    }
  }
};

} // namespace detail


// ---------------------------------------------------------------------------


} // namespace dray

#endif // DRAY_POS_SIMPLEX_ELEMENT_HPP

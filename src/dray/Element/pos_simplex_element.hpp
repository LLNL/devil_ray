#ifndef DRAY_POS_SIMPLEX_ELEMENT_HPP
#define DRAY_POS_SIMPLEX_ELEMENT_HPP

/**
 * @file pos_simplex_element.hpp
 * @brief Partial template specialization of Element_impl
 *        for simplex (i.e. tet and tri) elements.
 */

#include <dray/Element/element.hpp>
#include <dray/integer_utils.hpp>   // MultinomialCoeff
#include <dray/exports.hpp>
#include <dray/vec.hpp>

namespace dray
{
  //
  // TriElement_impl
  //
  template <typename T, uint32 dim, uint32 ncomp, int32 P>
  using TriElement_impl = Element_impl<T, dim, ncomp, ElemType::Tri, P>;


  template <typename T, uint32 dim>
  class TriRefSpace
  {
    public:
      DRAY_EXEC static bool is_inside(const Vec<T,dim> &ref_coords);  //TODO
      DRAY_EXEC static void clamp_to_domain(Vec<T,dim> &ref_coords);  //TODO
      DRAY_EXEC static Vec<T,dim> project_to_domain(const Vec<T,dim> &r1, const Vec<T,dim> &r2);  //TODO
  };

  template <uint32 dim>
  struct RefTri
  {
    Vec<float32, dim> m_vertices[dim + 1];

    DRAY_EXEC static RefTri ref_universe()
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

    DRAY_EXEC Vec<float32, dim> center() const
    {
      Vec<float32, dim> c;   c = 0.0;
      for (int d = 0; d <= dim; d++)
        c += m_vertices[d];
      c *= float32(1.0/(dim+1));
      return c;
    }

    DRAY_EXEC float32 max_length() const
    {
      // Any proxy for diameter. In this case use maximum edge length.
      float32 M = 0.0;
      for (int32 v1 = 0; v1 <= dim; v1++)
        for (int32 v2 = 0; v2 <= dim; v2++)
          M = fmaxf(M, (m_vertices[v1] - m_vertices[v2]).magnitude2());
      return sqrtf(M);
    }
  };

  // Specialize SubRef for Tri type.
  template <>
  struct ElemTypeAttributes<ElemType::Tri>
  {
    template <uint32 dim>
    using SubRef = RefTri<dim>;
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
   *           7  []   \`           v^2 * {  Mck(2;0,2,0)*u^0 }
   *          /         \`
   *    'v'  /           \` (t=0)
   *        /  []    []   \`        v^1 * { [Mck(2;0,1,1)*u^0]*t + Mck(2;1,1,0)*u^1 }
   *       /               \`
   *      /                 \`
   *        []    []    []          v^0 * { [[Mck(2;0,0,2)*u^0]*t + Mck(2;1,0,1)*u^1]*t + Mck(2;2,0,0)*u^2 }
   *  (t=1)
   *        ------------->
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
  template <typename T, uint32 ncomp>
  class Element_impl<T, 2u, ncomp, ElemType::Tri, Order::General> : public TriRefSpace<T,2u>
  {
    protected:
      SharedDofPtr<Vec<T, ncomp>> m_dof_ptr;
      uint32 m_order;
    public:
      DRAY_EXEC void construct(SharedDofPtr<Vec<T, ncomp>> dof_ptr, int32 poly_order)
      {
        m_dof_ptr = dof_ptr;
        m_order = poly_order;
      }
      DRAY_EXEC int32 get_order() const    { return m_order; }
      DRAY_EXEC int32 get_num_dofs() const { return get_num_dofs(m_order); }
      DRAY_EXEC static constexpr int32 get_num_dofs(int32 order) { return (order+1)*(order+2)/2; }

      DRAY_EXEC Vec<T, ncomp> eval(const Vec<T,2u> &ref_coords) const;

      DRAY_EXEC Vec<T, ncomp> eval_d( const Vec<T,2u> &ref_coords,
                                      Vec<Vec<T,ncomp>,2u> &out_derivs) const;

      DRAY_EXEC void get_sub_bounds(const RefTri<2u> &sub_ref, AABB<ncomp> &aabb) const;
  };



  // Template specialization (Tri type, general order, 3D).
  //
  template <typename T, uint32 ncomp>
  class Element_impl<T, 3u, ncomp, ElemType::Tri, Order::General> : public TriRefSpace<T,3u>
  {
    protected:
      SharedDofPtr<Vec<T, ncomp>> m_dof_ptr;
      uint32 m_order;
    public:
      DRAY_EXEC void construct(SharedDofPtr<Vec<T, ncomp>> dof_ptr, int32 poly_order)
      {
        m_dof_ptr = dof_ptr;
        m_order = poly_order;
      }
      DRAY_EXEC int32 get_order() const    { return m_order; }
      DRAY_EXEC int32 get_num_dofs() const { return get_num_dofs(m_order); }
      DRAY_EXEC static constexpr int32 get_num_dofs(int32 order) { return (order+1)*(order+2)*(order+3)/6; }

      DRAY_EXEC Vec<T, ncomp> eval(const Vec<T,3u> &ref_coords) const;

      DRAY_EXEC Vec<T, ncomp> eval_d( const Vec<T,3u> &ref_coords,
                                      Vec<Vec<T,ncomp>,3u> &out_derivs) const;

      DRAY_EXEC void get_sub_bounds(const RefTri<3u> &sub_ref, AABB<ncomp> &aabb) const;
  };


  // -----
  // Implementations
  // -----

  template <typename T, uint32 dim>
  DRAY_EXEC bool TriRefSpace<T,dim>::is_inside(const Vec<T,dim> &ref_coords)
  {
    T min_val = 2.f;
    T t = 1.0f;
    for (int32 d = 0; d < dim; d++)
    {
      min_val = min(ref_coords[d], min_val);
      t -= ref_coords[d];
    }
    min_val = min(t, min_val);
    return (min_val >= 0.f - epsilon<T>());
  }

  template <typename T, uint32 dim>
  DRAY_EXEC void TriRefSpace<T,dim>::clamp_to_domain(Vec<T,dim> &ref_coords)
  {
    //TODO
  }

  template <typename T, uint32 dim>
  DRAY_EXEC Vec<T,dim> TriRefSpace<T,dim>::project_to_domain(const Vec<T,dim> &r1, const Vec<T,dim> &r2)
  {
    return {0.0}; //TODO
  }

  // ------------


  // :: 2D :: //

  //
  // eval() (2D triangle evaluation)
  //
  template <typename T, uint32 ncomp>
  DRAY_EXEC Vec<T, ncomp>
  Element_impl<T, 2u, ncomp, ElemType::Tri, Order::General>::eval(const Vec<T,2u> &ref_coords) const
  {
    using DofT = Vec<T, ncomp>;
    using PtrT = SharedDofPtr<Vec<T, ncomp>>;

    const uint32 p = m_order;
    PtrT dof_ptr = m_dof_ptr;  // Make a local copy that can be incremented.

    // Barycentric coordinates.
    const T &u = ref_coords[0];
    const T &v = ref_coords[1];
    const T t = T(1.0) - (u + v);

    // Multinomial coefficient. Will traverse Pascal's simplex using
    // transitions between adjacent multinomial coefficients (slide_over()),
    // and transpositions back to the start of each row (swap_places()).
    MultinomialCoeff<2> mck;
    mck.construct(p);

    DofT j_sum; j_sum = 0.0;
    T vpow = 1.0;
    for (int32 jj = 0; jj <= p; jj++)
    {

      DofT i_sum; i_sum = 0.0;
      T upow = 1.0;
      for (int32 ii = 0; ii <= (p-jj); ii++)
      {
        // Horner's rule innermost, due to decreasing powers of t (mu = p - jj - ii).
        i_sum *= t;
        i_sum = i_sum + (*(dof_ptr)) * (mck.get_val() * upow);
        ++dof_ptr;
        upow *= u;
        if (ii < (p-jj))
          mck.slide_over(0);
      }
      mck.swap_places(0);

      j_sum = j_sum + i_sum * vpow;
      vpow *= v;
      if (jj < p)
        mck.slide_over(1);
    }
    //mck.swap_places(1);

    return j_sum;
  }


  //
  // eval_d() (2D triangle eval & derivatives)
  //
  template <typename T, uint32 ncomp>
  DRAY_EXEC Vec<T, ncomp>
  Element_impl<T, 2u, ncomp, ElemType::Tri, Order::General>::eval_d(
      const Vec<T,2u> &ref_coords,
      Vec<Vec<T,ncomp>,2u> &out_derivs) const
  {
    using DofT = Vec<T, ncomp>;
    using PtrT = SharedDofPtr<Vec<T, ncomp>>;

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
    PtrT dof_ptr = m_dof_ptr;  // Make a local copy that can be incremented.

    // Barycentric coordinates.
    const T &u = ref_coords[0];
    const T &v = ref_coords[1];
    const T t = T(1.0) - (u + v);

    // Multinomial coefficient. Will traverse Pascal's simplex using
    // transitions between adjacent multinomial coefficients (slide_over()),
    // and transpositions back to the start of each row (swap_places()).
    MultinomialCoeff<2> mck;
    mck.construct(pm1);

    int32 dof_idx = 0;

    DofT j_sum; j_sum = 0.0;
    Vec<DofT,2u> j_sum_d; j_sum_d = 0.0;
    T vpow = 1.0;
    for (int32 jj = 0; jj <= pm1; jj++)
    {
      const int32 sz_p_j = (p-jj + 1)/1;       // nchoosek(p-jj + dim-1, dim-1)

      DofT i_sum; i_sum = 0.0;
      Vec<DofT,2u> i_sum_d; i_sum_d = 0.0;
      T upow = 1.0;
      for (int32 ii = 0; ii <= (pm1-jj); ii++)
      {
        // Horner's rule innermost, due to decreasing powers of t (mu = pm1 - jj - ii).
        i_sum *= t;
        i_sum_d[0] *= t;
        i_sum_d[1] *= t;

        const DofT dof_mu = dof_ptr[dof_idx];
        const Vec<DofT,2u> dof_ij = { dof_ptr[dof_idx + 1],          // Offset dofs
                                      dof_ptr[dof_idx + sz_p_j] };
        dof_idx++;

        i_sum += (dof_mu*t + dof_ij[0]*u + dof_ij[1]*v) * (mck.get_val() * upow);
        i_sum_d[0] +=              (dof_ij[0] - dof_mu) * (mck.get_val() * upow);
        i_sum_d[1] +=              (dof_ij[1] - dof_mu) * (mck.get_val() * upow);

        upow *= u;
        if (ii < (pm1-jj))
          mck.slide_over(0);
      }
      mck.swap_places(0);

      dof_idx++;  // Skip end of row.

      j_sum += i_sum * vpow;
      j_sum_d += i_sum_d * vpow;
      vpow *= v;
      if (jj < pm1)
        mck.slide_over(1);
    }
    //mck.swap_places(1);

    out_derivs = j_sum_d * p;
    return j_sum;
  }


  template <typename T, uint32 ncomp>
  DRAY_EXEC void
  Element_impl<T, 2u, ncomp, ElemType::Tri, Order::General>::get_sub_bounds(
      const RefTri<2u> &sub_ref,
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
    // NOTE: This will defeat subdivision searches. It will cause the search space to
    // increase rather than decrease on each step.

    aabb.reset();
    const int num_dofs = get_num_dofs();
    for (int ii = 0; ii < num_dofs; ii++)
      aabb.include(m_dof_ptr[ii]);
  }


  // :: 3D :: //

  //
  // eval() (3D tetrahedron evaluation)
  //
  template <typename T, uint32 ncomp>
  DRAY_EXEC Vec<T, ncomp>
  Element_impl<T, 3u, ncomp, ElemType::Tri, Order::General>::eval(const Vec<T,3u> &ref_coords) const
  {
    using DofT = Vec<T, ncomp>;
    using PtrT = SharedDofPtr<Vec<T, ncomp>>;

    const unsigned int p = m_order;
    PtrT dof_ptr = m_dof_ptr;  // Make a local copy that can be incremented.

    // Barycentric coordinates.
    const T &u = ref_coords[0];
    const T &v = ref_coords[1];
    const T &w = ref_coords[2];
    const T t = T(1.0) - (u + v + w);

    // Multinomial coefficient. Will traverse Pascal's simplex using
    // transitions between adjacent multinomial coefficients (slide_over()),
    // and transpositions back to the start of each row (swap_places()).
    MultinomialCoeff<3> mck;
    mck.construct(p);

    DofT k_sum; k_sum = 0.0;
    T wpow = 1.0;
    for (int32 kk = 0; kk <= p; kk++)
    {

      DofT j_sum; j_sum = 0.0;
      T vpow = 1.0;
      for (int32 jj = 0; jj <= p-kk; jj++)
      {

        DofT i_sum; i_sum = 0.0;
        T upow = 1.0;
        for (int32 ii = 0; ii <= (p-kk-jj); ii++)
        {
          // Horner's rule innermost, due to decreasing powers of t (mu = p - kk - jj - ii).
          i_sum *= t;
          i_sum += (*dof_ptr) * (mck.get_val() * upow);
          ++dof_ptr;
          upow *= u;
          if (ii < (p-kk-jj))
            mck.slide_over(0);
        }
        mck.swap_places(0);

        j_sum += i_sum * vpow;
        vpow *= v;
        if (jj < p-kk)
          mck.slide_over(1);
      }
      mck.swap_places(1);

      k_sum += j_sum * wpow;
      wpow *= w;
      if (kk < p)
        mck.slide_over(2);
    }
    // mck.swap_places(2);

    return k_sum;
  }


  //
  // eval_d() (3D tetrahedron eval & derivatives)
  //
  template <typename T, uint32 ncomp>
  DRAY_EXEC Vec<T, ncomp>
  Element_impl<T, 3u, ncomp, ElemType::Tri, Order::General>::eval_d(
      const Vec<T,3u> &ref_coords,
      Vec<Vec<T,ncomp>,3u> &out_derivs) const
  {
    using DofT = Vec<T, ncomp>;
    using PtrT = SharedDofPtr<Vec<T, ncomp>>;

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
    PtrT dof_ptr = m_dof_ptr;  // Make a local copy that can be incremented.

    // Barycentric coordinates.
    const T &u = ref_coords[0];
    const T &v = ref_coords[1];
    const T &w = ref_coords[2];
    const T t = T(1.0) - (u + v + w);

    // Multinomial coefficient. Will traverse Pascal's simplex using
    // transitions between adjacent multinomial coefficients (slide_over()),
    // and transpositions back to the start of each row (swap_places()).
    MultinomialCoeff<3> mck;
    mck.construct(pm1);

    int32 dof_idx = 0;

    DofT k_sum;  k_sum = 0.0;
    Vec<DofT,3u> k_sum_d;  k_sum_d = 0.0;
    T wpow = 1.0;
    for (int32 kk = 0; kk <= pm1; kk++)
    {
      const int32 sz_p_k = (p-kk + 1)*(p-kk + 2)/(1*2);         // nchoosek(p-kk + dim-1, dim-1)

      DofT j_sum; j_sum = 0.0;
      Vec<DofT,3u> j_sum_d;  j_sum_d = 0.0;
      T vpow = 1.0;
      for (int32 jj = 0; jj <= (pm1-kk); jj++)
      {
        const int32 sz_p_j = (p-kk-jj + 1)/1;       // nchoosek(q-jj + dim-2, dim-2)

        DofT i_sum; i_sum = 0.0;
        Vec<DofT,3u> i_sum_d;  i_sum_d = 0.0;
        T upow = 1.0;
        for (int32 ii = 0; ii <= (pm1-kk-jj); ii++)
        {
          // Horner's rule innermost, due to decreasing powers of t (mu = pm1 - kk - jj - ii).
          i_sum *= t;
          i_sum_d[0] *= t;
          i_sum_d[1] *= t;
          i_sum_d[2] *= t;

          const DofT dof_mu = dof_ptr[dof_idx];
          const Vec<DofT,3u> dof_ijk = { dof_ptr[dof_idx + 1],         // Offset dofs
                                         dof_ptr[dof_idx + sz_p_j],
                                         dof_ptr[dof_idx + sz_p_k - jj] };
          dof_idx++;

          i_sum += (dof_mu*t + dof_ijk[0]*u + dof_ijk[1]*v + dof_ijk[2]*w) * (mck.get_val() * upow);
          i_sum_d[0] +=                              (dof_ijk[0] - dof_mu) * (mck.get_val() * upow);
          i_sum_d[1] +=                              (dof_ijk[1] - dof_mu) * (mck.get_val() * upow);
          i_sum_d[2] +=                              (dof_ijk[2] - dof_mu) * (mck.get_val() * upow);

          upow *= u;
          if (ii < (pm1-kk-jj))
            mck.slide_over(0);
        }
        mck.swap_places(0);

        dof_idx++;  // Skip end of row.

        j_sum += i_sum * vpow;
        j_sum_d += i_sum_d * vpow;
        vpow *= v;
        if (jj < (pm1-kk))
          mck.slide_over(1);
      }
      mck.swap_places(1);

      dof_idx++;  // Skip tip of triangle.

      k_sum += j_sum * wpow;
      k_sum_d += j_sum_d * wpow;
      wpow *= w;
      if (kk < pm1)
        mck.slide_over(2);
    }
    //mck.swap_places(2);

    out_derivs = k_sum_d * p;
    return k_sum;
  }


  template <typename T, uint32 ncomp>
  DRAY_EXEC void
  Element_impl<T, 3u, ncomp, ElemType::Tri, Order::General>::get_sub_bounds(
      const RefTri<3u> &sub_ref,
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
    // NOTE: This will defeat subdivision searches. It will cause the search space to
    // increase rather than decrease on each step.

    aabb.reset();
    const int num_dofs = get_num_dofs();
    for (int ii = 0; ii < num_dofs; ii++)
      aabb.include(m_dof_ptr[ii]);
  }




  // ---------------------------------------------------------------------------


}//namespace dray

#endif// DRAY_POS_SIMPLEX_ELEMENT_HPP

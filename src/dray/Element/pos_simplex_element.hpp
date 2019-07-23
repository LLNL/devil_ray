#ifndef DRAY_POS_SIMPLEX_ELEMENT_HPP
#define DRAY_POS_SIMPLEX_ELEMENT_HPP

/**
 * @file pos_simplex_element.hpp
 * @brief Partial template specialization of Element_impl
 *        for simplex (i.e. tet and tri) elements.
 */

#include <dray/Element/element.hpp>
#include <dray/integer_utils.hpp>   // MultinomialCoeff
#include <dray/vec.hpp>

namespace dray
{
namespace newelement
{
  //
  // TriElement_impl
  //
  template <typename T, uint32 dim, int32 P>
  using TriElement_impl = Element_impl<T, dim, ElemType::Tri, P>;

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
  template <typename T>
  class Element_impl<T, 2u, ElemType::Tri, Order::General>
  {
    protected:
      uint32 m_order;
    public:
      void construct(int32 poly_order) { m_order = poly_order; }
      int32 get_order() const          { return m_order; }
      int32 get_num_dofs() const       { return (m_order+1)*(m_order+2)/2; }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC DofT eval(const Vec<T,2u> &ref_coords, PtrT dof_ptr)
      {
        const uint32 p = m_order;

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
            i_sum = i_sum + (*(dof_ptr++)) * (mck.get_val() * upow);
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


      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC DofT eval_d( const Vec<T,2u> &ref_coords,
                                    PtrT dof_ptr,
                                    Vec<DofT,2u> &out_derivs)
      {
        if (m_order == 0)
        {
          out_derivs[0] = 0.0;
          out_derivs[1] = 0.0;
          return dof_ptr[0];
        }

        // The Bernstein--Bezier simplex basis has the following properties:
        //
        // - Derivatives in terms of (p-1)-order triangle:
        //     du = \sum_{i + j + \mu = p-1} \beta^{p-1}_{i, j, \mu} \left( C_{i+1, j, \mu} - C_{i, j, \mu+1} \right)
        //     dv = \sum_{i + j + \mu = p-1} \beta^{p-1}_{i, j, \mu} \left( C_{i, j+1, \mu} - C_{i, j, \mu+1} \right)
        //
        // - p-order triangle in terms of (p-1)-order triangle:
        //     F = \sum_{i + j + \mu = p-1} \beta^{p-1}_{i, j, \mu} \left( C_{i+1, j, \mu} + C_{i, j+1, \mu} + C_{i, j, \mu+1} \right)

        // The dof offset in each axis depends on the index in that axis and greater axes.
        // We can compute the offsets by noticing the hierarchical structure of simplex volumes.
        // A slice of the (d)-dim, (p)-order simplex at index q is a (d-1)-dim, (p-q)-order simplex.
        //     #DOFS(p, dim) = \sum_{q=0}^{p} #DOFS(q, dim-1) = nchoosek(p+dim, dim)
        //     Offset(q) = #DOFS(q, dim-1).

        const uint32 p = m_order;
        const uint32 pm1 = m_order - 1;

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
          const int32 sz_pm1_j = (pm1-jj + 1)/1;   // nchoosek(p-1-jj + dim-1, dim-1)

          DofT i_sum; i_sum = 0.0;
          Vec<DofT,2u> i_sum_d; i_sum_d = 0.0;
          T upow = 1.0;
          for (int32 ii = 0; ii <= (pm1-jj); ii++)
          {
            // Horner's rule innermost, due to decreasing powers of t (mu = pm1 - jj - ii).
            i_sum *= t;
            i_sum_d *= t;

            const DofT dof_mu = dof_ptr[dof_idx];
            const Vec<DofT,2u> dof_ij = { dof_ptr[dof_idx + 1],
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

          dof_ptr += (sz_p_j - sz_pm1_j);

          j_sum = j_sum + i_sum * vpow;
          j_sum_d = j_sum_d + i_sum_d * vpow;
          vpow *= v;
          if (jj < pm1)
            mck.slide_over(1);
        }
        //mck.swap_places(1);

        out_derivs = j_sum_d * p;
        return j_sum;
      }
  };



  // Template specialization (Tri type, general order, 3D).
  //
  template <typename T>
  class Element_impl<T, 3u, ElemType::Tri, Order::General>
  {
    protected:
      uint32 m_order;
    public:
      void construct(int32 poly_order) { m_order = poly_order; }
      int32 get_order() const          { return m_order; }
      int32 get_num_dofs() const       { return (m_order+1)*(m_order+2)*(m_order+3)/6; }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC DofT eval(const Vec<T,3u> &ref_coords, PtrT dof_ptr)
      {
        const unsigned int p = m_order;

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
              i_sum = i_sum + (*(dof_ptr++)) * (mck.get_val() * upow);
              upow *= u;
              if (ii < (p-kk-jj))
                mck.slide_over(0);
            }
            mck.swap_places(0);

            j_sum = j_sum + i_sum * vpow;
            vpow *= v;
            if (jj < p-kk)
              mck.slide_over(1);
          }
          mck.swap_places(1);
        
          k_sum = k_sum + j_sum * wpow;
          wpow *= w;
          if (kk < p)
            mck.slide_over(2);
        }
        // mck.swap_places(2);

        return k_sum;
      }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC DofT eval_d( const Vec<T,3u> &ref_coords,
                                    PtrT dof_ptr,
                                    Vec<DofT,3u> &out_derivs)
      {
        //TODO
        DofT answer;
        answer = 0;
        return answer;
      }
  };




  // ---------------------------------------------------------------------------


}//namespace newelement
}//namespace dray

#endif// DRAY_POS_SIMPLEX_ELEMENT_HPP

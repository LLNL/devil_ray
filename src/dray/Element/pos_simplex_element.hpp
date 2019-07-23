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
   * Barycentric coordinates u + v + t = 1.
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
        const unsigned int p = m_order;

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
        //TODO
        DofT answer;
        answer = 0;
        return answer;
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

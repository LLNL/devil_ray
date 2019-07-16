#ifndef DRAY_BEZIER_SIMPLEX_HPP
#define DRAY_BEZIER_SIMPLEX_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

//TODO put into a new multinomial class file.
template <int32 dim>
class MultinomialCoeff
{
  // Invariant: m_val = MultinomialCoeff(n; i, j, k).
  // Invariant: i+j+k = n.
  protected:
    int32 m_val;
    int32 m_n;
    int32 m_ijk[dim + 1];

  public:
    // Before using MultinomialCoeff, call construct(n).
    DRAY_EXEC void construct(int32 n)
    {
      constexpr int32 full_place = dim;
      m_val = 1;
      m_n = n;
      for (int32 d = 0; d <= dim; d++)
        m_ijk[d] = 0;
      m_ijk[full_place] = n;
    }

    // Getters.
    DRAY_EXEC int32        get_val() const { return m_val; }
    DRAY_EXEC int32        get_n()   const { return m_n; }
    DRAY_EXEC const int32 *get_ijk() const { return m_ijk; }

    // slice_over() - Advance to next coefficient along a direction.
    //                Be careful not to slide off Pascal's simplex.
    DRAY_EXEC int32 slide_over(int32 inc_place)
    {
      constexpr int32 dec_place = dim;
      //       n!              n!         k
      // ---------------  =  -------  *  ---
      // (i+1)! M (k-1)!     i! M k!      i
      /// if (m_ijk[dec_place])
      m_val *= m_ijk[dec_place];
      if (m_ijk[inc_place])
        m_val /= m_ijk[inc_place];
      m_ijk[dec_place]--;
      m_ijk[inc_place]++;
      return m_val;
    }

    // swap_places() - The multinomial coefficient is symmetric in i, j, k.
    DRAY_EXEC void swap_places(int32 place1, int32 place2 = dim)
    {
      const int32 s = m_ijk[place2];
      m_ijk[place2] = m_ijk[place1];
      m_ijk[place1] = s;
    }
};


template <typename T, int32 dim>
struct BezierSimplex
{
  //TODO
};



/*
 * Example
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


//
// BezierSimplex<T,2> (Bezier triangle).
//
template <typename T>
struct BezierSimplex<T,2>
{
  // eval(): Evaluation without computing derivatives.
  template <typename DofT, typename PtrT = const DofT*>
  DRAY_EXEC static DofT eval(const Vec<T,2> &ref_coords, PtrT dof_ptr, const int32 p)
  {
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

};


//
// BezierSimplex<T,3> (Bezier tetrahedron).
//
template <typename T>
struct BezierSimplex<T,3>
{
  //TODO
};


}//namespace dray


#endif//DRAY_BEZIER_SIMPLEX_HPP

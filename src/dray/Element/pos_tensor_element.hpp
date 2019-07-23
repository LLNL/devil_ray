#ifndef DRAY_POS_TENSOR_ELEMENT_HPP
#define DRAY_POS_TENSOR_ELEMENT_HPP

/**
 * @file pos_tensor_element.hpp
 * @brief Partial template specialization of Element_impl
 *        for tensor (i.e. hex and quad) elements.
 */

#include <dray/Element/element.hpp>
#include <dray/integer_utils.hpp>   // MultinomialCoeff
#include <dray/vec.hpp>

namespace dray
{
namespace newelement
{
  //
  // QuadElement_impl
  //
  template <typename T, uint32 dim, int32 P>
  using QuadElement_impl = Element_impl<T, dim, ElemType::Quad, P>;


  // ---------------------------------------------------------------------------

  // Template specialization (Quad type, general order).
  //
  // Assume dim <= 3.
  //
  template <typename T, uint32 dim>
  class Element_impl<T, dim, ElemType::Quad, Order::General>
  {
    protected:
      uint32 m_order;
    public:
      void construct(int32 poly_order) { m_order = poly_order; }
      int32 get_order() const          { return m_order; }
      int32 get_num_dofs() const       { return intPow(m_order+1, dim); }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC DofT eval(const Vec<T,dim> &r, PtrT dof_ptr)
      {
        //TODO
        DofT answer; answer = 0;
        return answer;
      }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC DofT eval_d( const Vec<T,dim> &ref_coords,
                                    PtrT dof_ptr,
                                    Vec<DofT,dim> &out_derivs)
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

        DofT zero;
        zero = 0;

        const T u = (dim > 0 ? ref_coords[0] : 0.0);
        const T v = (dim > 1 ? ref_coords[1] : 0.0);
        const T w = (dim > 2 ? ref_coords[2] : 0.0);
        const T ubar = 1.0 - u;
        const T vbar = 1.0 - v;
        const T wbar = 1.0 - w;

        const int32 p1 = (dim >= 1 ? m_order : 0);
        const int32 p2 = (dim >= 2 ? m_order : 0);
        const int32 p3 = (dim >= 3 ? m_order : 0);

        int32 B[MaxPolyOrder];
        if (m_order >= 1)
        {
          BinomialCoeff binomial_coeff;
          binomial_coeff.construct(m_order - 1);
          for (int32 ii = 0; ii <= m_order - 1; ii++)
          {
            B[ii] = binomial_coeff.get_val();
            binomial_coeff.slide_over(0);
          }
        }

        int32 cidx = 0;  // Index into element dof indexing space.

        // Compute and combine order (p-1) values to get order (p) values/derivatives.
        // https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties

        DofT val_u, val_v, val_w;
        DofT        deriv_u;
        Vec<DofT,2> deriv_uv;
        Vec<DofT,3> deriv_uvw;

        // Level3 set up.
        T wpow = 1.0;
        Vec<DofT,3> val_w_L, val_w_R;  // Second/third columns are derivatives in lower level.
        val_w_L = zero;
        val_w_R = zero;
        for (int32 ii = 0; ii <= p3; ii++)
        {
          // Level2 set up.
          T vpow = 1.0;
          Vec<DofT,2> val_v_L, val_v_R;  // Second column is derivative in lower level.
          val_v_L = zero;
          val_v_R = zero;

          for (int32 jj = 0; jj <= p2; jj++)
          {
            // Level1 set up.
            T upow = 1.0;
            DofT val_u_L = zero, val_u_R = zero;           // L and R can be combined --> val, deriv.
            DofT C = dof_ptr[cidx++];
            for (int32 kk = 1; kk <= p1; kk++)
            {
              // Level1 accumulation.
              val_u_L = val_u_L * ubar + C * (B[kk-1] * upow);
              C = dof_ptr[cidx++];
              val_u_R = val_u_R * ubar + C * (B[kk-1] * upow);
              upow *= u;
            }//kk

            // Level1 result.
            val_u = (p1 > 0 ? val_u_L * ubar + val_u_R * u : C);
            deriv_u = (val_u_R - val_u_L) * p1;

            // Level2 accumulation.
            if (jj > 0)
            {
              val_v_R[0] = val_v_R[0] * vbar + val_u   * (B[jj-1] * vpow);
              val_v_R[1] = val_v_R[1] * vbar + deriv_u * (B[jj-1] * vpow);
              vpow *= v;
            }
            if (jj < p2)
            {
              val_v_L[0] = val_v_L[0] * vbar + val_u   * (B[jj] * vpow);
              val_v_L[1] = val_v_L[1] * vbar + deriv_u * (B[jj] * vpow);
            }
          }//jj

          // Level2 result.
          val_v       = (p2 > 0 ? val_v_L[0] * vbar + val_v_R[0] * v : val_u);
          deriv_uv[0] = (p2 > 0 ? val_v_L[1] * vbar + val_v_R[1] * v : deriv_u);
          deriv_uv[1] = (val_v_R[0] - val_v_L[0]) * p2;

          // Level3 accumulation.
          if (ii > 0)
          {
            val_w_R[0] = val_w_R[0] * wbar + val_v       * (B[ii-1] * wpow);
            val_w_R[1] = val_w_R[1] * wbar + deriv_uv[0] * (B[ii-1] * wpow);
            val_w_R[2] = val_w_R[2] * wbar + deriv_uv[1] * (B[ii-1] * wpow);
            wpow *= w;
          }
          if (ii < p3)
          {
            val_w_L[0] = val_w_L[0] * wbar + val_v       * (B[ii] * wpow);
            val_w_L[1] = val_w_L[1] * wbar + deriv_uv[0] * (B[ii] * wpow);
            val_w_L[2] = val_w_L[2] * wbar + deriv_uv[1] * (B[ii] * wpow);
          }
        }//ii

        // Level3 result.
        val_w        = (p3 > 0 ? val_w_L[0] * wbar + val_w_R[0] * w : val_v);
        deriv_uvw[0] = (p3 > 0 ? val_w_L[1] * wbar + val_w_R[1] * w : deriv_uv[0]);
        deriv_uvw[1] = (p3 > 0 ? val_w_L[2] * wbar + val_w_R[2] * w : deriv_uv[1]);
        deriv_uvw[2] = (val_w_R[0] - val_w_L[0]) * p3;

        if (dim > 0) out_derivs[0] = deriv_uvw[0];
        if (dim > 1) out_derivs[1] = deriv_uvw[1];
        if (dim > 2) out_derivs[2] = deriv_uvw[2];

        return val_w;
      }
  };


  // ---------------------------------------------------------------------------


  // Template specialization (Tensor type, 0th order).
  //
  template <typename T, uint32 dim>
  class Element_impl<T, dim, ElemType::Quad, Order::Constant>
  {
    public:
      void construct() {}
      void construct(int32) {}
      static constexpr int32 get_order() { return 0; }
      static constexpr int32 get_num_dofs() { return 1; }

      // Get value without derivative.
      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval(const Vec<T,dim> &ref_coords, PtrT dof_ptr)
      {
        return *dof_ptr;
      }

      // Get value with derivative.
      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval_d( const Vec<T,dim> &ref_coords,
                                    PtrT dof_ptr,
                                    Vec<DofT,dim> &out_derivs)
      {
        for (int d = 0; d < dim; d++)
          out_derivs[d] = 0;

        return *dof_ptr;
      }
  };


  // Template specialization (Quad type, 1st order, 2D).
  //
  template <typename T>
  class Element_impl<T, 2u, ElemType::Quad, Order::Linear>
  {
    public:
      void construct() {}
      void construct(int32) {}
      static constexpr int32 get_order() { return 1; }
      static constexpr int32 get_num_dofs() { return 4; }

      // Get value without derivative.
      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval(const Vec<T,2u> &r, PtrT dof_ptr)
      {
        return dof_ptr[0] * (1-r[0]) * (1-r[1]) +
               dof_ptr[1] *    r[0]  * (1-r[1]) +
               dof_ptr[2] * (1-r[0]) *    r[1]  +
               dof_ptr[3] *    r[0]  *    r[1];
      }

      // Get value with derivative.
      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval_d( const Vec<T,2u> &r,
                                    PtrT dof_ptr,
                                    Vec<DofT,2u> &out_derivs)
      {
        out_derivs[0] = (dof_ptr[1] - dof_ptr[0]) * (1-r[1]) +
                        (dof_ptr[3] - dof_ptr[2]) *    r[1];

        out_derivs[1] = (dof_ptr[2] - dof_ptr[0]) * (1-r[0]) +
                        (dof_ptr[3] - dof_ptr[1]) *    r[0];

        return dof_ptr[0] * (1-r[0]) * (1-r[1]) +
               dof_ptr[1] *    r[0]  * (1-r[1]) +
               dof_ptr[2] * (1-r[0]) *    r[1]  +
               dof_ptr[3] *    r[0]  *    r[1];
      }
  };


  // Template specialization (Quad type, 1st order, 3D).
  //
  template <typename T>
  class Element_impl<T, 3u, ElemType::Quad, Order::Linear>
  {
    public:
      void construct() {}
      void construct(int32) {}
      static constexpr int32 get_order() { return 1; }
      static constexpr int32 get_num_dofs() { return 8; }

      // Get value without derivative.
      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval(const Vec<T,3u> &r, PtrT dof_ptr)
      {
        return dof_ptr[0] * (1-r[0]) * (1-r[1]) * (1-r[2]) +
               dof_ptr[1] *    r[0]  * (1-r[1]) * (1-r[2]) +
               dof_ptr[2] * (1-r[0]) *    r[1]  * (1-r[2]) +
               dof_ptr[3] *    r[0]  *    r[1]  * (1-r[2]) +
               dof_ptr[4] * (1-r[0]) * (1-r[1]) *    r[2]  +
               dof_ptr[5] *    r[0]  * (1-r[1]) *    r[2]  +
               dof_ptr[6] * (1-r[0]) *    r[1]  *    r[2]  +
               dof_ptr[7] *    r[0]  *    r[1]  *    r[2]  ;
      }

      // Get value with derivative.
      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval_d( const Vec<T,3u> &r,
                                    PtrT dof_ptr,
                                    Vec<DofT,3u> &out_derivs)
      {
        out_derivs[0] = (dof_ptr[1] - dof_ptr[0]) * (1-r[1]) * (1-r[2]) +
                        (dof_ptr[3] - dof_ptr[2]) *    r[1]  * (1-r[2]) +
                        (dof_ptr[5] - dof_ptr[4]) * (1-r[1]) *    r[2]  +
                        (dof_ptr[7] - dof_ptr[6]) *    r[1]  *    r[2]  ;

        out_derivs[1] = (dof_ptr[2] - dof_ptr[0]) * (1-r[0]) * (1-r[2]) +
                        (dof_ptr[3] - dof_ptr[1]) *    r[0]  * (1-r[2]) +
                        (dof_ptr[6] - dof_ptr[4]) * (1-r[0]) *    r[2]  +
                        (dof_ptr[7] - dof_ptr[5]) *    r[0]  *    r[2]  ;

        out_derivs[2] = (dof_ptr[4] - dof_ptr[0]) * (1-r[0]) * (1-r[1]) +
                        (dof_ptr[5] - dof_ptr[1]) *    r[0]  * (1-r[1]) +
                        (dof_ptr[6] - dof_ptr[2]) * (1-r[0]) *    r[1]  +
                        (dof_ptr[7] - dof_ptr[3]) *    r[0]  *    r[1]  ;

        return dof_ptr[0] * (1-r[0]) * (1-r[1]) * (1-r[2]) +
               dof_ptr[1] *    r[0]  * (1-r[1]) * (1-r[2]) +
               dof_ptr[2] * (1-r[0]) *    r[1]  * (1-r[2]) +
               dof_ptr[3] *    r[0]  *    r[1]  * (1-r[2]) +
               dof_ptr[4] * (1-r[0]) * (1-r[1]) *    r[2]  +
               dof_ptr[5] *    r[0]  * (1-r[1]) *    r[2]  +
               dof_ptr[6] * (1-r[0]) *    r[1]  *    r[2]  +
               dof_ptr[7] *    r[0]  *    r[1]  *    r[2]  ;
      }
  };





  // Template specialization (Quad type, 2nd order, 2D).
  //
  template <typename T>
  class Element_impl<T, 2u, ElemType::Quad, Order::Quadratic>
  {
    public:
      void construct() {}
      void construct(int32) {}
      static constexpr int32 get_order() { return 2; }
      static constexpr int32 get_num_dofs() { return IntPow<3,2u>::val; }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval(const Vec<T,2u> &r, PtrT dof_ptr)
      {
        // Shape functions. Quadratic has 3 1D shape functions on each axis.
        T su[3] = { (1-r[0])*(1-r[0]),  2*r[0]*(1-r[0]),  r[0]*r[0] };
        T sv[3] = { (1-r[1])*(1-r[1]),  2*r[1]*(1-r[1]),  r[1]*r[1] };

        return dof_ptr[0] * su[0] * sv[0] +
               dof_ptr[1] * su[1] * sv[0] +
               dof_ptr[2] * su[2] * sv[0] +
               dof_ptr[3] * su[0] * sv[1] +
               dof_ptr[4] * su[1] * sv[1] +
               dof_ptr[5] * su[2] * sv[1] +
               dof_ptr[6] * su[0] * sv[2] +
               dof_ptr[7] * su[1] * sv[2] +
               dof_ptr[8] * su[2] * sv[2] ;
      }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval_d( const Vec<T,2u> &r,
                                    PtrT dof_ptr,
                                    Vec<DofT,2u> &out_derivs)
      {
        // Shape functions. Quadratic has 3 1D shape functions on each axis.
        T su[3] = { (1-r[0])*(1-r[0]),  2*r[0]*(1-r[0]),  r[0]*r[0] };
        T sv[3] = { (1-r[1])*(1-r[1]),  2*r[1]*(1-r[1]),  r[1]*r[1] };

        // Shape derivatives.
        T dsu[3] = { -1+r[0],  1-r[0]-r[0],  r[0] };
        T dsv[3] = { -1+r[1],  1-r[1]-r[1],  r[1] };

        out_derivs[0] = dof_ptr[0] * dsu[0] * sv[0] +
                        dof_ptr[1] * dsu[1] * sv[0] +
                        dof_ptr[2] * dsu[2] * sv[0] +
                        dof_ptr[3] * dsu[0] * sv[1] +
                        dof_ptr[4] * dsu[1] * sv[1] +
                        dof_ptr[5] * dsu[2] * sv[1] +
                        dof_ptr[6] * dsu[0] * sv[2] +
                        dof_ptr[7] * dsu[1] * sv[2] +
                        dof_ptr[8] * dsu[2] * sv[2] ;

        out_derivs[1] = dof_ptr[0] * su[0] * dsv[0] +
                        dof_ptr[1] * su[1] * dsv[0] +
                        dof_ptr[2] * su[2] * dsv[0] +
                        dof_ptr[3] * su[0] * dsv[1] +
                        dof_ptr[4] * su[1] * dsv[1] +
                        dof_ptr[5] * su[2] * dsv[1] +
                        dof_ptr[6] * su[0] * dsv[2] +
                        dof_ptr[7] * su[1] * dsv[2] +
                        dof_ptr[8] * su[2] * dsv[2] ;

        return dof_ptr[0] * su[0] * sv[0] +
               dof_ptr[1] * su[1] * sv[0] +
               dof_ptr[2] * su[2] * sv[0] +
               dof_ptr[3] * su[0] * sv[1] +
               dof_ptr[4] * su[1] * sv[1] +
               dof_ptr[5] * su[2] * sv[1] +
               dof_ptr[6] * su[0] * sv[2] +
               dof_ptr[7] * su[1] * sv[2] +
               dof_ptr[8] * su[2] * sv[2] ;
      }
  };


  // Template specialization (Quad type, 2nd order, 3D).
  //
  template <typename T>
  class Element_impl<T, 3u, ElemType::Quad, Order::Quadratic>
  {
    public:
      void construct() {}
      void construct(int32) {}
      static constexpr int32 get_order() { return 2; }
      static constexpr int32 get_num_dofs() { return IntPow<3,3u>::val; }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval(const Vec<T,3u> &r, PtrT dof_ptr)
      {
        //TODO
        DofT answer; answer = 0;
        return answer;
      }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval_d( const Vec<T,3u> &r,
                                    PtrT dof_ptr,
                                    Vec<DofT,3u> &out_derivs)
      {
        // Shape functions. Quadratic has 3 1D shape functions on each axis.
        T su[3] = { (1-r[0])*(1-r[0]),  2*r[0]*(1-r[0]),  r[0]*r[0] };
        T sv[3] = { (1-r[1])*(1-r[1]),  2*r[1]*(1-r[1]),  r[1]*r[1] };
        T sw[3] = { (1-r[2])*(1-r[2]),  2*r[2]*(1-r[2]),  r[2]*r[2] };

        // Shape derivatives.
        T dsu[3] = { -1+r[0],  1-r[0]-r[0],  r[0] };
        T dsv[3] = { -1+r[1],  1-r[1]-r[1],  r[1] };
        T dsw[3] = { -1+r[2],  1-r[2]-r[2],  r[2] };

        out_derivs[0] =  dof_ptr[0]  * dsu[0] * sv[0] * sw[0] +
                         dof_ptr[1]  * dsu[1] * sv[0] * sw[0] +
                         dof_ptr[2]  * dsu[2] * sv[0] * sw[0] +
                         dof_ptr[3]  * dsu[0] * sv[1] * sw[0] +
                         dof_ptr[4]  * dsu[1] * sv[1] * sw[0] +
                         dof_ptr[5]  * dsu[2] * sv[1] * sw[0] +
                         dof_ptr[6]  * dsu[0] * sv[2] * sw[0] +
                         dof_ptr[7]  * dsu[1] * sv[2] * sw[0] +
                         dof_ptr[8]  * dsu[2] * sv[2] * sw[0] +

                         dof_ptr[9]  * dsu[0] * sv[0] * sw[1] +
                         dof_ptr[10] * dsu[1] * sv[0] * sw[1] +
                         dof_ptr[11] * dsu[2] * sv[0] * sw[1] +
                         dof_ptr[12] * dsu[0] * sv[1] * sw[1] +
                         dof_ptr[13] * dsu[1] * sv[1] * sw[1] +
                         dof_ptr[14] * dsu[2] * sv[1] * sw[1] +
                         dof_ptr[15] * dsu[0] * sv[2] * sw[1] +
                         dof_ptr[16] * dsu[1] * sv[2] * sw[1] +
                         dof_ptr[17] * dsu[2] * sv[2] * sw[1] +

                         dof_ptr[18] * dsu[0] * sv[0] * sw[2] +
                         dof_ptr[19] * dsu[1] * sv[0] * sw[2] +
                         dof_ptr[20] * dsu[2] * sv[0] * sw[2] +
                         dof_ptr[21] * dsu[0] * sv[1] * sw[2] +
                         dof_ptr[22] * dsu[1] * sv[1] * sw[2] +
                         dof_ptr[23] * dsu[2] * sv[1] * sw[2] +
                         dof_ptr[24] * dsu[0] * sv[2] * sw[2] +
                         dof_ptr[25] * dsu[1] * sv[2] * sw[2] +
                         dof_ptr[26] * dsu[2] * sv[2] * sw[2] ;

        out_derivs[1] =  dof_ptr[0]  * su[0] * dsv[0] * sw[0] +
                         dof_ptr[1]  * su[1] * dsv[0] * sw[0] +
                         dof_ptr[2]  * su[2] * dsv[0] * sw[0] +
                         dof_ptr[3]  * su[0] * dsv[1] * sw[0] +
                         dof_ptr[4]  * su[1] * dsv[1] * sw[0] +
                         dof_ptr[5]  * su[2] * dsv[1] * sw[0] +
                         dof_ptr[6]  * su[0] * dsv[2] * sw[0] +
                         dof_ptr[7]  * su[1] * dsv[2] * sw[0] +
                         dof_ptr[8]  * su[2] * dsv[2] * sw[0] +

                         dof_ptr[9]  * su[0] * dsv[0] * sw[1] +
                         dof_ptr[10] * su[1] * dsv[0] * sw[1] +
                         dof_ptr[11] * su[2] * dsv[0] * sw[1] +
                         dof_ptr[12] * su[0] * dsv[1] * sw[1] +
                         dof_ptr[13] * su[1] * dsv[1] * sw[1] +
                         dof_ptr[14] * su[2] * dsv[1] * sw[1] +
                         dof_ptr[15] * su[0] * dsv[2] * sw[1] +
                         dof_ptr[16] * su[1] * dsv[2] * sw[1] +
                         dof_ptr[17] * su[2] * dsv[2] * sw[1] +

                         dof_ptr[18] * su[0] * dsv[0] * sw[2] +
                         dof_ptr[19] * su[1] * dsv[0] * sw[2] +
                         dof_ptr[20] * su[2] * dsv[0] * sw[2] +
                         dof_ptr[21] * su[0] * dsv[1] * sw[2] +
                         dof_ptr[22] * su[1] * dsv[1] * sw[2] +
                         dof_ptr[23] * su[2] * dsv[1] * sw[2] +
                         dof_ptr[24] * su[0] * dsv[2] * sw[2] +
                         dof_ptr[25] * su[1] * dsv[2] * sw[2] +
                         dof_ptr[26] * su[2] * dsv[2] * sw[2] ;

        out_derivs[2] =  dof_ptr[0]  * su[0] * sv[0] * dsw[0] +
                         dof_ptr[1]  * su[1] * sv[0] * dsw[0] +
                         dof_ptr[2]  * su[2] * sv[0] * dsw[0] +
                         dof_ptr[3]  * su[0] * sv[1] * dsw[0] +
                         dof_ptr[4]  * su[1] * sv[1] * dsw[0] +
                         dof_ptr[5]  * su[2] * sv[1] * dsw[0] +
                         dof_ptr[6]  * su[0] * sv[2] * dsw[0] +
                         dof_ptr[7]  * su[1] * sv[2] * dsw[0] +
                         dof_ptr[8]  * su[2] * sv[2] * dsw[0] +

                         dof_ptr[9]  * su[0] * sv[0] * dsw[1] +
                         dof_ptr[10] * su[1] * sv[0] * dsw[1] +
                         dof_ptr[11] * su[2] * sv[0] * dsw[1] +
                         dof_ptr[12] * su[0] * sv[1] * dsw[1] +
                         dof_ptr[13] * su[1] * sv[1] * dsw[1] +
                         dof_ptr[14] * su[2] * sv[1] * dsw[1] +
                         dof_ptr[15] * su[0] * sv[2] * dsw[1] +
                         dof_ptr[16] * su[1] * sv[2] * dsw[1] +
                         dof_ptr[17] * su[2] * sv[2] * dsw[1] +

                         dof_ptr[18] * su[0] * sv[0] * dsw[2] +
                         dof_ptr[19] * su[1] * sv[0] * dsw[2] +
                         dof_ptr[20] * su[2] * sv[0] * dsw[2] +
                         dof_ptr[21] * su[0] * sv[1] * dsw[2] +
                         dof_ptr[22] * su[1] * sv[1] * dsw[2] +
                         dof_ptr[23] * su[2] * sv[1] * dsw[2] +
                         dof_ptr[24] * su[0] * sv[2] * dsw[2] +
                         dof_ptr[25] * su[1] * sv[2] * dsw[2] +
                         dof_ptr[26] * su[2] * sv[2] * dsw[2] ;

        return dof_ptr[0]  * su[0] * sv[0] * sw[0] +
               dof_ptr[1]  * su[1] * sv[0] * sw[0] +
               dof_ptr[2]  * su[2] * sv[0] * sw[0] +
               dof_ptr[3]  * su[0] * sv[1] * sw[0] +
               dof_ptr[4]  * su[1] * sv[1] * sw[0] +
               dof_ptr[5]  * su[2] * sv[1] * sw[0] +
               dof_ptr[6]  * su[0] * sv[2] * sw[0] +
               dof_ptr[7]  * su[1] * sv[2] * sw[0] +
               dof_ptr[8]  * su[2] * sv[2] * sw[0] +

               dof_ptr[9]  * su[0] * sv[0] * sw[1] +
               dof_ptr[10] * su[1] * sv[0] * sw[1] +
               dof_ptr[11] * su[2] * sv[0] * sw[1] +
               dof_ptr[12] * su[0] * sv[1] * sw[1] +
               dof_ptr[13] * su[1] * sv[1] * sw[1] +
               dof_ptr[14] * su[2] * sv[1] * sw[1] +
               dof_ptr[15] * su[0] * sv[2] * sw[1] +
               dof_ptr[16] * su[1] * sv[2] * sw[1] +
               dof_ptr[17] * su[2] * sv[2] * sw[1] +

               dof_ptr[18] * su[0] * sv[0] * sw[2] +
               dof_ptr[19] * su[1] * sv[0] * sw[2] +
               dof_ptr[20] * su[2] * sv[0] * sw[2] +
               dof_ptr[21] * su[0] * sv[1] * sw[2] +
               dof_ptr[22] * su[1] * sv[1] * sw[2] +
               dof_ptr[23] * su[2] * sv[1] * sw[2] +
               dof_ptr[24] * su[0] * sv[2] * sw[2] +
               dof_ptr[25] * su[1] * sv[2] * sw[2] +
               dof_ptr[26] * su[2] * sv[2] * sw[2] ;
      }
  };



  // Template specialization (Quad type, 3rd order).
  //
  template <typename T, uint32 dim>
  class Element_impl<T, dim, ElemType::Quad, Order::Cubic>
  {
    public:
      void construct() {}
      void construct(int32) {}
      static constexpr int32 get_order() { return 3; }
      static constexpr int32 get_num_dofs() { return IntPow<4,dim>::val; }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval(const Vec<T,dim> &r, PtrT dof_ptr)
      {
        //TODO
        DofT answer; answer = 0;
        return answer;
      }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval_d( const Vec<T,dim> &ref_coords,
                                    PtrT dof_ptr,
                                    Vec<DofT,dim> &out_derivs)
      {
        //TODO
        DofT answer; answer = 0;
        return answer;
      }
  };




}//namespace newelement
}//namespace dray

#endif// DRAY_POS_TENSOR_ELEMENT_HPP

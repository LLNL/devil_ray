#ifndef DRAY_POS_TENSOR_ELEMENT_HPP
#define DRAY_POS_TENSOR_ELEMENT_HPP

/**
 * @file pos_tensor_element.hpp
 * @brief Partial template specialization of Element_impl
 *        for tensor (i.e. hex and quad) elements.
 */

#include <dray/Element/element.hpp>
#include <dray/vec.hpp>

namespace dray
{
namespace newelement
{
  //
  // QuadElement_impl
  //
  template <typename T, uint32 dim, Order P>
  using QuadElement_impl = Element_impl<T, dim, ElemType::Quad, P>;


  // Template specialization (Tensor type, 0th order).
  //
  template <typename T, uint32 dim>
  class Element_impl<T, dim, ElemType::Quad, Order::Constant>
  {
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





  // Template specialization (Quad type, 2nd order).
  //
  template <typename T, uint32 dim>
  class Element_impl<T, dim, ElemType::Quad, Order::Quadratic>
  {
    void construct() {}
    void construct(int32) {}
    static constexpr int32 get_order() { return 2; }
    static constexpr int32 get_num_dofs() { return IntPow<3,dim>::val; }

    template <typename DofT, typename PtrT = const DofT*>
    DRAY_EXEC static DofT eval(const Vec<T,3u> &r, PtrT dof_ptr)
    {
      //TODO
      DofT answer; answer = 0;
      return answer;
    }

    template <typename DofT, typename PtrT = const DofT*>
    DRAY_EXEC static DofT eval_d( const Vec<T,3u> &ref_coords,
                                  PtrT dof_ptr,
                                  Vec<DofT,3u> &out_derivs)
    {
      //TODO
      DofT answer; answer = 0;
      return answer;
    }
  };


  // Template specialization (Quad type, 3rd order).
  //
  template <typename T, uint32 dim>
  class Element_impl<T, dim, ElemType::Quad, Order::Cubic>
  {
    void construct() {}
    void construct(int32) {}
    static constexpr int32 get_order() { return 3; }
    static constexpr int32 get_num_dofs() { return IntPow<4,dim>::val; }

    template <typename DofT, typename PtrT = const DofT*>
    DRAY_EXEC static DofT eval(const Vec<T,3u> &r, PtrT dof_ptr)
    {
      //TODO
      DofT answer; answer = 0;
      return answer;
    }

    template <typename DofT, typename PtrT = const DofT*>
    DRAY_EXEC static DofT eval_d( const Vec<T,3u> &ref_coords,
                                  PtrT dof_ptr,
                                  Vec<DofT,3u> &out_derivs)
    {
      //TODO
      DofT answer; answer = 0;
      return answer;
    }
  };


  // Template specialization (Quad type, general order).
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
      DRAY_EXEC static DofT eval(const Vec<T,3u> &r, PtrT dof_ptr)
      {
        //TODO
        DofT answer; answer = 0;
        return answer;
      }

      template <typename DofT, typename PtrT = const DofT*>
      DRAY_EXEC static DofT eval_d( const Vec<T,3u> &ref_coords,
                                    PtrT dof_ptr,
                                    Vec<DofT,3u> &out_derivs)
      {
        //TODO
        DofT answer; answer = 0;
        return answer;
      }
  };





}//namespace newelement
}//namespace dray

#endif// DRAY_POS_TENSOR_ELEMENT_HPP

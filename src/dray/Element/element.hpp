#ifndef DRAY_ELEMENT_HPP
#define DRAY_ELEMENT_HPP

#include <dray/el_trans.hpp>
#include <dray/bernstein_basis.hpp>
#include <dray/vec.hpp>
#include <dray/range.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>

namespace dray
{

  // (2019-05-23) TODO right now this simply hides the ugliness of ElTransOp and family.
  //   Eventually this class should take the place of ElTransOp.
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  using ElementBase = ElTransOp<T, BernsteinBasis<T,RefDim>, ElTransIter<T,PhysDim>>;

  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  class Element
  {
    public:
      using Base = ElementBase<T,RefDim,PhysDim>;
      static constexpr int32 phys_dim = PhysDim;    // Seen by NewtonSolve
      static constexpr int32 ref_dim = RefDim;      // Seen by NewtonSolve

      //
      // create() : factory method.
      DRAY_EXEC static Element create(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr, T *aux_mem_ptr);
      
      //
      // construct() : constructor you must call explicitly.
      DRAY_EXEC void construct(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr, T *aux_mem_ptr);

      //
      // get_bounds()
      DRAY_EXEC void get_bounds(Range *ranges) const;

      //TODO a set of evaluator functions, v, v_d, pv, jac. For now use ElTransOp::eval().

      DRAY_EXEC void eval(const Vec<T,RefDim> &ref, Vec<T,PhysDim> &result_val,
                          Vec<Vec<T,PhysDim>,RefDim> &result_deriv) const
      {
        m_base.eval(ref, result_val, result_deriv);
      }


      //
      // is_inside()
      DRAY_EXEC static bool is_inside(const Vec<T,RefDim> &ref_coords);

      //
      // get_el_id()
      DRAY_EXEC int32 get_el_id() const { return m_el_id; }

    protected:
      int32 m_el_id;
      mutable Base m_base;  // Composition rather than public/private inheritance, because const means different things.
  };


}



// Implementations
namespace dray
{

  //
  // create() : factory method.
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC Element<T,RefDim,PhysDim> Element<T,RefDim,PhysDim>::create(int32 el_id, int32 poly_order,
      const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr, T *aux_mem_ptr)
  {
    Element<T,RefDim,PhysDim> ret;
    ret.construct(el_id, poly_order, ctrl_idx_ptr, val_ptr, aux_mem_ptr);
    return ret;
  }

  //
  // construct() : constructor you must call explicitly.
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC void Element<T,RefDim,PhysDim>::construct(
      int32 el_id, int32 poly_order,
      const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr, T *aux_mem_ptr)
  {
    /// BernsteinBasis<T,RefDim>::init_shape(poly_order, aux_mem_ptr);
    /// Base::m_coeff_iter.init_iter(ctrl_idx_ptr, val_ptr, intPow(poly_order+1, RefDim), el_id);

    m_base.init_shape(poly_order, aux_mem_ptr);
    m_base.m_coeff_iter.init_iter(ctrl_idx_ptr, val_ptr, intPow(poly_order+1, RefDim), el_id);
    m_el_id = el_id;
  }

  //
  // get_bounds()
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC void Element<T,RefDim,PhysDim>::get_bounds(Range *ranges) const
  {
    ElTransData<T, PhysDim>::get_elt_node_range(m_base.m_coeff_iter, m_base.get_el_dofs(), ranges);
  }


  //
  // is_inside()
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC bool Element<T,RefDim,PhysDim>::is_inside(const Vec<T,RefDim> &ref_coords)
  {
    for (int32 d = 0; d < RefDim; d++)
      if (!(0.0 <= ref_coords[0] && ref_coords[0] < 1.0))
        return false;
    return true;
  }
}

#endif//DRAY_ELEMENT_HPP


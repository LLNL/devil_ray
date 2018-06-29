#include <dray/high_order_shape.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/arrayvec.hpp>
#include <dray/types.hpp>

#include <assert.h>
#include <iostream>
#include <stdio.h>

namespace dray
{

///template <typename T, int32 RefDim>
///DRAY_EXEC
///void BernsteinShape<T,RefDim>::calc_shape_dshape_1d(
///    const int32 p, const T x, const T y,
///    T *u, T *d)

template <typename T, int32 RefDim, typename Shape1D>
void TensorShape<T, RefDim, Shape1D>::calc_shape_dshape(
    const Array<int32> &active_idx,
    const ArrayVec<T,RefDim> &ref_pts,
    Array<T> &shape_val,                          // Will be resized.
    ArrayVec<T,RefDim> &shape_deriv) const        // Will be resized.
{
  //
  // This method uses the virtual method el_dofs_1d().
  //

    // - Use size_queries, q_idx to index into ref_pts.
    // - Use size_active, a_q_idx to index into active_idx, shape_va, shape_deriv.
  const int32 size_queries = ref_pts.size();
  const int32 size_active = active_idx.size();
  const int32 el_dofs_1d = get_el_dofs_1d();      // Virtual method.
  const int32 el_dofs = get_el_dofs();            // Class method, but uses virtual get_el_dofs_1d().

  // Intermediate output arrays for 1D polynomials.
  Array<T> shape_val_1d;          // 1D eval for each ref variable, stored as XXXYYYZZZ|XXXYYYZZZ|...
  Array<T> shape_deriv_1d;        // 1D deriv eval for each ref variable, stored as XXXYYYZZZ|XXXYYYZZZ|...
  shape_val_1d.resize(el_dofs_1d * RefDim * size_active);
  shape_deriv_1d.resize(el_dofs_1d * RefDim * size_active);

  // Compute intermediate outputs.
  // The 1D polynomial for each variable can be evaluated at each ref_pt independently.
  T *val_1d_ptr = shape_val_1d.get_device_ptr();
  T *deriv_1d_ptr = shape_deriv_1d.get_device_ptr();
  const int32 *active_idx_ptr = active_idx.get_device_ptr_const();
  const ScalarVec<T,RefDim> *ref_ptr = ref_pts.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, RefDim * size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 rdim = aii % RefDim;
    const int32 a_q_idx = aii / RefDim;
    const int32 q_idx = active_idx_ptr[a_q_idx];

    const T ref_coord = *( (T*) &ref_ptr[q_idx] + rdim );   // ref_pts[q_idx][rdim], or ref_pts[q_idx][0] if scalar.

    const int32 out_offset = el_dofs_1d * aii;
    Shape1D::calc_shape_dshape_1d(el_dofs_1d,
        ref_coord, 1-ref_coord,
        val_1d_ptr + out_offset,
        deriv_1d_ptr + out_offset);
  });

  // Create new output arrays shape_val and shape_deriv, of size_active.
  shape_val.resize(el_dofs * size_active);
  shape_deriv.resize(el_dofs * size_active);

  // Compute and store the tensor product of the 1D arrays.
  // Each component of the tensor product can be evaluated at each ref_pt independently.
  T *val_ptr = shape_val.get_device_ptr();
  ScalarVec<T,RefDim> *deriv_ptr = shape_deriv.get_device_ptr();
  TensorProduct<T,RefDim,1,1> tensor_product_val;
  TensorProduct<T,RefDim,1,RefDim> tensor_product_deriv;
  
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, el_dofs * size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 tens_dim = aii % el_dofs;
    const int32 a_q_idx = aii / el_dofs;

    //TODO I think the bug is somewhere in this indexing madness.

    // Calculate the start of each 1D polynomial eval array for the current q_idx.
    const T* val_starts[RefDim];
    const T* deriv_starts[RefDim];
    for (int32 rdim = 0; rdim < RefDim; rdim++)
    {
      int32 offset_1d = el_dofs_1d * (rdim + RefDim * a_q_idx);
      val_starts[rdim] = val_1d_ptr + offset_1d;
      deriv_starts[rdim] = deriv_1d_ptr + offset_1d;
    }

    const int32 offset_q = el_dofs * a_q_idx;

    // Compute the output value by taking tensor product of all value rows.
    tensor_product_val(el_dofs_1d, val_starts, tens_dim, val_ptr + offset_q);

    // Compute each partial derivative by taking the tensor product
    // of that 1D derivative with the values of the other 1D rows.
    // Each partial derivative is a component in a Vec<RefDim>, so
    // the output stride is RefDim.
    for (int32 rdim = 0; rdim < RefDim; rdim++)
    {
      // Substitute 1D derivative in place of value, for the rdim- partial derivative.
      const T* const swap_val_start = val_starts[rdim];
      val_starts[rdim] = deriv_starts[rdim];

      T* const deriv_comp_ptr = (T *) (deriv_ptr + offset_q) + rdim;

      tensor_product_deriv(el_dofs_1d, val_starts, tens_dim, deriv_comp_ptr);

      // Undo the initial substitution.
      val_starts[rdim] = swap_val_start;
    }
  });
}

// Explicit instantiations.
template class TensorShape<float32, 3, Linear1D<float32>>;
template class TensorShape<float64, 3, Linear1D<float64>>;
template class TensorShape<float32, 3, Bernstein1D<float32>>;
template class TensorShape<float64, 3, Bernstein1D<float64>>;
template class TensorShape<float32, 2, Bernstein1D<float32>>;    //Debug
template class TensorShape<float64, 2, Bernstein1D<float64>>;    //Debug
template class TensorShape<float32, 1, Bernstein1D<float32>>;    //Debug
template class TensorShape<float64, 1, Bernstein1D<float64>>;    //Debug



  // Set the given attributes, and resize arrays.
template <typename T, int32 P, int32 R, typename ST>
void
ElTrans<T,P,R,ST>::resize(int32 size_el, int32 el_dofs, ST shape, int32 size_ctrl)
{
  // Check that el_dofs and RefDim are consistent with shape.
  assert( el_dofs == shape.get_el_dofs() );
  assert( C_RefDim == shape.get_ref_dim() );

  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_ctrl;
  m_shape = shape;
  
  m_ctrl_idx.resize(size_el * el_dofs);
  m_values.resize(size_ctrl);
}

  // This method assumes that output arrays are already the right size.
  // It does not resize or assign new arrays to output parameters.
template <typename T, int32 P, int32 R, typename ST>
void
ElTrans<T,P,R,ST>::eval(const Array<int> &active_idx,
            const Array<int32> &el_ids, const ArrayVec<T,R> &ref_pts,
            ArrayVec<T,P> &trans_val, Array<Matrix<T,P,R>> &trans_deriv) const
{
  const int32 size_queries = ref_pts.size();
  const int32 size_active = active_idx.size();

  // Evaluate shape at all active reference point.
  Array<T> shape_val;
  ArrayVec<T,C_RefDim> shape_deriv;
  m_shape.calc_shape_dshape(active_idx, ref_pts, shape_val, shape_deriv);
    // Now shape_val and shape_deriv have been resized according to active_idx.

  //DEBUG
  ////std::cout << "shape_val ";     shape_val.summary();
  ////std::cout << "m_ctrl_idx ";    m_ctrl_idx.summary();
  ////std::cout << "m_values ";      m_values.summary();


  // Intermediate data.
  const T *shape_val_ptr = shape_val.get_device_ptr_const();
  const ScalarVec<T,C_RefDim> *shape_deriv_ptr = shape_deriv.get_device_ptr_const();
 
  // Member data.
  const ScalarVec<T,C_PhysDim> *values_ptr = m_values.get_device_ptr_const();
  const int32 *ctrl_idx_ptr = m_ctrl_idx.get_device_ptr_const();
  const int32 el_dofs = m_el_dofs;   // local for lambda capture.

  // Output data.
  ScalarVec<T,C_PhysDim> *trans_val_ptr = trans_val.get_device_ptr();
  Matrix<T,C_PhysDim,C_RefDim> *trans_deriv_ptr = trans_deriv.get_device_ptr();

  // Input data.
  const int32 *active_idx_ptr = active_idx.get_device_ptr_const();
  const int32 *el_ids_ptr = el_ids.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
  //for (int32 aii = 0; aii < size_active; aii++)
  {
    const int32 q_idx = active_idx_ptr[aii];
    const int32 el_idx = el_ids_ptr[q_idx];

    //No longer needed.
    ///    // Compute element shape values.
    ///    ShapeVec elt_shape;
    ///    shape_f.calc_shape(ref_pts_ptr[elt_idx], elt_shape);
    ///    //shape_f(ref_pts_ptr[elt_idx], elt_shape);

    ////std::cout << "(dof,ci,shape,cv):   ";

    // Grab and accumulate the control point values for this element,
    // weighted by the shape values.
    // (This part is sequential and not parallel because it is a "segmented reduction.")
    ScalarVec<T,C_PhysDim> elt_val;
    elt_val = static_cast<T>(0.f);
    //Matrix    //TODO accumulator for matrix.
    for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
    {
      ////elt_val += elt_shape[dof_idx] * values_ptr[ctrl_idx_ptr[elt_idx*DOF + dof_idx]];
      ////elt_val += shape_val_ptr[aii*el_dofs + dof_idx] * values_ptr[ctrl_idx_ptr[el_idx*el_dofs + dof_idx]];
      const T sv = shape_val_ptr[aii * el_dofs + dof_idx];
      const int32 ci = ctrl_idx_ptr[el_idx * el_dofs + dof_idx];
      const ScalarVec<T,C_PhysDim> cv = values_ptr[ci];
      elt_val += cv * sv;

      //DEBUG
      ////if (abs(sv) > 0.05)
      ////{
      ////  printf("(%d, %d, %.2f, %.1f) ", dof_idx, ci,sv,*(T*)&cv);
      ////}
    }

    ////std::cout << std::endl;

    trans_val_ptr[q_idx] = elt_val;
  //}
  });
}

// Explicit instantiations.
template class ElTrans<float32, 1, 3, BernsteinShape<float32, 3>>;  //e.g. scalar field
template class ElTrans<float64, 1, 3, BernsteinShape<float64, 3>>;

template class ElTrans<float32, 3, 1, BernsteinShape<float32, 1>>;  //e.g. ray in space
template class ElTrans<float64, 3, 1, BernsteinShape<float64, 1>>;

template class ElTrans<float32, 3, 3, BernsteinShape<float32, 3>>;  //e.g. high-order geometry
template class ElTrans<float64, 3, 3, BernsteinShape<float64, 3>>;

template class ElTrans<float32, 4, 4, BernsteinShape<float32, 4>>;
template class ElTrans<float64, 4, 4, BernsteinShape<float64, 4>>;

///   // Clients may read and write contents of member arrays, but not size of member arrays.
/// template <typename T, int32 P, int32 R, typename ST>
/// const int32 *
/// ElTrans<T,P,R,ST>::ctrl_idx_get_host_ptr_const() const
/// { return m_ctrl_idx.get_host_ptr_const(); }
/// 
/// template <typename T, int32 P, int32 R, typename ST>
/// const int32 *
/// ElTrans<T,P,R,ST>::ctrl_idx_get_device_ptr_const() const
/// { return m_ctrl_idx.get_device_ptr_const(); }
/// 
/// template <typename T, int32 P, int32 R, typename ST>
/// int32 *
/// ElTrans<T,P,R,ST>::ctrl_idx_get_host_ptr()
/// { return m_ctrl_idx.get_host_ptr(); }
/// 
/// template <typename T, int32 P, int32 R, typename ST>
/// int32 *
/// ElTrans<T,P,R,ST>::ctrl_idx_get_device_ptr()
/// { return m_ctrl_idx.get_device_ptr(); }
/// 
/// template <typename T, int32 P, int32 R, typename ST>
/// const int32 *
/// ElTrans<T,P,R,ST>::values_get_host_ptr_const() const
/// { return m_values.get_host_ptr_const(); }
/// 
/// template <typename T, int32 P, int32 R, typename ST>
/// const int32 *
/// ElTrans<T,P,R,ST>::values_get_device_ptr_const() const
/// { return m_values.get_device_ptr_const(); }
/// 
/// template <typename T, int32 P, int32 R, typename ST>
/// int32 *
/// ElTrans<T,P,R,ST>::values_get_host_ptr()
/// { return m_values.get_host_ptr(); }
/// 
/// template <typename T, int32 P, int32 R, typename ST>
/// int32 *
/// ElTrans<T,P,R,ST>::values_get_device_ptr()
/// { return m_values.get_device_ptr(); }




/*  ----


template <typename T, int32 C, int32 DOF, typename ShapeFunctor, int32 D>
ArrayVec<T,C>
FunctionCtrlPoints<T,C,DOF, ShapeFunctor,D>::eval(const ShapeFunctor &_shape_f, const ArrayVec<T,D> &ref_pts) const
{
  /// // Check that shape_f has the right dimensions (compile time).
  /// const ShapeDims<D,DOF> cmpl_test_shape_dims = ShapeFunctor::shape_dims;

  const int32 num_elts = m_ctrl_idx.size() / DOF;
  assert(ref_pts.size() == num_elts);

  const ShapeFunctor shape_f = _shape_f;  // local for lambda capture.
  const RefVec *ref_pts_ptr = ref_pts.get_device_ptr_const();
  const PhysVec *values_ptr = m_values.get_device_ptr_const();
  const int32 *ctrl_idx_ptr = m_ctrl_idx.get_device_ptr_const();

  Array<PhysVec> elt_vals_out;
  elt_vals_out.resize(num_elts);
  PhysVec *elt_vals_ptr = elt_vals_out.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elts), [=] DRAY_LAMBDA (int32 elt_idx)
  {
    // Compute element shape values.
    ShapeVec elt_shape;
    shape_f.calc_shape(ref_pts_ptr[elt_idx], elt_shape);
    //shape_f(ref_pts_ptr[elt_idx], elt_shape);

    // Grab and accumulate the control point values for this element,
    // weighted by the shape values.
    // (This part is sequential and not parallel because it is a "segmented reduction.")
    PhysVec elt_val = static_cast<T>(0.f);
    for (int32 dof_idx = 0; dof_idx < DOF; dof_idx++)
    {
      elt_val += elt_shape[dof_idx] * values_ptr[ctrl_idx_ptr[elt_idx*DOF + dof_idx]];
    }

    elt_vals_ptr[elt_idx] = elt_val;
  });

  return elt_vals_out;
}

---- */

// Explicit instantiations.

//template class FunctionCtrlPoints<float32, 1,27, detail::DummyUniformShape<float32,3,27>, 3>;

} // namespace dray

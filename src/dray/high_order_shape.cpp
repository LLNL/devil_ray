#include <dray/high_order_shape.hpp>
#include <dray/array_utils.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>

#include <assert.h>
#include <iostream>
#include <stdio.h>

namespace dray
{

template <typename T, int32 RefDim, typename Shape1D>
void TensorShape<T, RefDim, Shape1D>::calc_shape_dshape(
    const Array<int32> &active_idx,
    const Array<Vec<T,RefDim>> &ref_pts,
    Array<T> &shape_val,                          // Will be resized.
    Array<Vec<T,RefDim>> &shape_deriv) const        // Will be resized.
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
  const Vec<T,RefDim> *ref_ptr = ref_pts.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, RefDim * size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 rdim = aii % RefDim;
    const int32 a_q_idx = aii / RefDim;
    const int32 q_idx = active_idx_ptr[a_q_idx];

    const T ref_coord = ref_ptr[q_idx][rdim];

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
  Vec<T,RefDim> *deriv_ptr = shape_deriv.get_device_ptr();
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
  assert( RefDim == shape.get_ref_dim() );

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
            const Array<int32> &el_ids, const Array<Vec<T,R>> &ref_pts,
            Array<Vec<T,P>> &trans_val, Array<Matrix<T,P,R>> &trans_deriv) const
{
  const int32 size_queries = ref_pts.size();
  const int32 size_active = active_idx.size();

  // Evaluate shape at all active reference point.
  Array<T> shape_val;
  Array<Vec<T,RefDim>> shape_deriv;
  m_shape.calc_shape_dshape(active_idx, ref_pts, shape_val, shape_deriv);
    // Now shape_val and shape_deriv have been resized according to active_idx.

  // Intermediate data.
  const T *shape_val_ptr = shape_val.get_device_ptr_const();
  const Vec<T,RefDim> *shape_deriv_ptr = shape_deriv.get_device_ptr_const();
 
  // Member data.
  const Vec<T,PhysDim> *values_ptr = m_values.get_device_ptr_const();
  const int32 *ctrl_idx_ptr = m_ctrl_idx.get_device_ptr_const();
  const int32 el_dofs = m_el_dofs;   // local for lambda capture.

  // Output data.
  Vec<T,PhysDim> *trans_val_ptr = trans_val.get_device_ptr();
  Matrix<T,PhysDim,RefDim> *trans_deriv_ptr = trans_deriv.get_device_ptr();

  // Input data.
  const int32 *active_idx_ptr = active_idx.get_device_ptr_const();
  const int32 *el_ids_ptr = el_ids.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 q_idx = active_idx_ptr[aii];
    const int32 el_idx = el_ids_ptr[q_idx];

    // Grab and accumulate the control point values for this element,
    // weighted by the shape values.
    // (This part is sequential and not parallel because it is a "segmented reduction.")
    Vec<T,PhysDim> elt_val;
    elt_val = static_cast<T>(0.f);
    Matrix<T,PhysDim,RefDim> elt_deriv;
    elt_deriv = static_cast<T>(0.f);

    for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
    {
      // Get control point values.
      const int32 ci = ctrl_idx_ptr[el_idx * el_dofs + dof_idx];
      const Vec<T,PhysDim> cv = values_ptr[ci];

      // Shape values are weights for control point values -> element value.
      const T sv = shape_val_ptr[aii * el_dofs + dof_idx];
      elt_val += cv * sv;

      // Shape derivatives are weights for control point values -> element derivatives.
      const Vec<T,RefDim> sd = shape_deriv_ptr[aii * el_dofs + dof_idx];
      elt_deriv += Matrix<T,PhysDim,RefDim>::outer_product(cv, sd);
    }

    trans_val_ptr[q_idx] = elt_val;
    trans_deriv_ptr[q_idx] = elt_deriv;
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



// -----------------------
// Support for NewtonSolve
// -----------------------


template <typename QueryType>
int32 NewtonSolve<QueryType>::step(
    const Array<Vec<T,phys_dim>> &target,
    QueryType &q,
    const Array<int32> &query_active,
    int32 max_steps,
    Array<int32> &solve_status)
{
  //TODO make these as parameters somewhere
  constexpr T tol_phys = 0.0000001;
  constexpr T tol_ref = 0.0000001;


  const int32 size_query = q.size();
  const int32 size_active = query_active.size();

  solve_status.resize(size_active);   // Resize output.
  array_memset(solve_status, (int32) NotConverged);

  int32 num_not_convg = size_active;

  // The initial guess for each query is already loaded in query.m_ref_pts.

  int32 it = 0;
  do
  {
    // Compute Phi and Jacobian at the current guess.
    q.query(query_active);

    // Device pointers.
    int32 *solve_status_ptr = solve_status.get_device_ptr();
    const int32 *active_idx_ptr = query_active.get_device_ptr_const();
    const Vec<T,phys_dim> *target_ptr = target.get_device_ptr_const();

    typename QueryType::ptr_bundle_const_t val_ptrb = q.get_val_device_ptr_const();
    typename QueryType::ptr_bundle_const_t deriv_ptrb = q.get_deriv_device_ptr_const();
    typename QueryType::ptr_bundle_t ref_ptrb = q.get_ref_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
    {
      // Only proceed if not yet converged.
      if (solve_status_ptr[aii] == NotConverged)
      {
        const int32 q_idx = active_idx_ptr[aii];

        // Compute delta_y.
        Vec<T,phys_dim> delta_y = QueryType::get_val(val_ptrb, q_idx);
        //Vec<T,phys_dim> delta_y = QueryType::get_val<true>(val_ptrb, q_idx);
        delta_y = target_ptr[q_idx] - delta_y;

        // Check for convergence in physical coordinates.
        if (delta_y.Normlinf() < tol_phys)
        {
          solve_status_ptr[aii] = ConvergePhys;
          return;  // Skip the rest of the lambda.
        }

        // Optionally check boundary -- currently not.

        // Check iteration: Don't perform another Newton step if we have hit max_steps.
        if (it >= max_steps)
        {
          return;  // Skip the rest of the lambda.
        }

        // Perform a Newton step to get delta_x.
        Matrix<T,phys_dim,ref_dim> jacobian = QueryType::get_deriv(deriv_ptrb, q_idx);
        //Matrix<T,phys_dim,ref_dim> jacobian = QueryType::get_deriv<true>(deriv_ptrb, q_idx);
        bool inverse_valid;
        Vec<T,ref_dim> delta_x = matrix_mult_inv(jacobian, delta_y, inverse_valid);  //Compiler error if ref_dim != phys_dim.

        // Check for convergence in reference coordinates
        if (delta_x.Normlinf() < tol_ref)
        {
          solve_status_ptr[aii] = ConvergeRef;
          return;  // Skip the rest of the lambda.
        }

        // No convergence so far.
        // Update the reference coordinate by adding delta_x.
        // Continue iterating.
        if (inverse_valid)
        {
          Vec<T,ref_dim> new_ref = QueryType::get_ref(ref_ptrb, q_idx) + delta_x;
          //Vec<T,ref_dim> new_ref = QueryType::get_ref<false>(ref_ptrb, q_idx) + delta_x;
          QueryType::set_ref(ref_ptrb, q_idx, new_ref);
        }
      }
    });  // end RAJA Newton Step

    //TODO RAJA sum up num_not_convg.

  }
  while (it++ < max_steps && num_not_convg > 0);  // End outer iterations.

  return it;
}

// Explicit instantiations.
///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 1, 3>>>;  //e.g. scalar field
///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 1, 3>>>;

///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 3, 1>>>;  //e.g. ray in space
///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 3, 1>>>;

template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 3, 3>>>;  //e.g. high-order geometry
template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 3, 3>>>;

template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 4, 4>>>;
template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 4, 4>>>;


} // namespace dray

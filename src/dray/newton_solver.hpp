#ifndef DRAY_NEWTON_SOLVER_HPP
#define DRAY_NEWTON_SOLVER_HPP

namespace dray
{

template <typename T>
struct NewtonSolve
{
  enum SolveStatus
  {
    NotConverged = 0,
    ConvergePhys = 1,
    ConvergeRef = 2
  };

  // solve() - The element id is implicit in trans.m_coeff_iter.
  //           The "initial guess" ref pt is set by the caller in [in]/[out] param "ref".
  //           The returned solution ref pt is set by the function in [in]/[out] "ref".
  //
  template <class TransOpType>
  DRAY_EXEC static SolveStatus solve(
      TransOpType &trans,
      const Vec<T,TransOpType::phys_dim> &target, Vec<T,TransOpType::ref_dim> &ref,
      const T tol_phys, const T tol_ref,
      int32 &steps_taken, const int32 max_steps = 10);

    // A version that also keeps the result of the last evaluation.
  template <class TransOpType>
  DRAY_EXEC static SolveStatus solve(
      TransOpType &trans,
      const Vec<T,TransOpType::phys_dim> &target, Vec<T,TransOpType::ref_dim> &ref,
      Vec<T, TransOpType::phys_dim> &y,
      Vec<Vec<T, TransOpType::phys_dim>, TransOpType::ref_dim> &deriv_cols,
      const T tol_phys, const T tol_ref,
      int32 &steps_taken, const int32 max_steps = 10);
};

template <typename T>
  template <class TransOpType>
DRAY_EXEC typename NewtonSolve<T>::SolveStatus
NewtonSolve<T>::solve(
    TransOpType &trans,
    const Vec<T,TransOpType::phys_dim> &target,
    Vec<T,TransOpType::ref_dim> &ref,
    const T tol_phys,
    const T tol_ref,
    int32 &steps_taken,
    const int32 max_steps)
{
  constexpr int32 phys_dim = TransOpType::phys_dim;
  constexpr int32 ref_dim = TransOpType::ref_dim;
  Vec<T,phys_dim>               y;
  Vec<Vec<T,phys_dim>,ref_dim>  deriv_cols;

  return solve( trans, target, ref, y, deriv_cols, tol_phys, tol_ref, steps_taken, max_steps);
}


template <typename T>
  template <class TransOpType>
DRAY_EXEC typename NewtonSolve<T>::SolveStatus
NewtonSolve<T>::solve(
    TransOpType &trans,
    const Vec<T,TransOpType::phys_dim> &target,
    Vec<T,TransOpType::ref_dim> &ref,
    Vec<T, TransOpType::phys_dim> &y,
    Vec<Vec<T, TransOpType::phys_dim>, TransOpType::ref_dim> &deriv_cols,
    const T tol_phys,
    const T tol_ref,
    int32 &steps_taken,
    const int32 max_steps)
{
  // The element id is implicit in trans.m_coeff_iter.
  // The "initial guess" reference point is set in the [in]/[out] argument "ref".

  constexpr int32 phys_dim = TransOpType::phys_dim;
  constexpr int32 ref_dim = TransOpType::ref_dim;
  assert(phys_dim == ref_dim);   // Need square jacobian.

  Vec<T,ref_dim>                x = ref;
  //Vec<T,phys_dim>               y, delta_y;
  Vec<T,phys_dim>               delta_y;
  //Vec<Vec<T,phys_dim>,ref_dim>  deriv_cols;

  NewtonSolve<T>::SolveStatus convergence_status;  // return value.

  // Evaluate at current ref pt and measure physical error.
  trans.eval(x, y, deriv_cols);
  delta_y = target - y;
  convergence_status = (delta_y.Normlinf() < tol_phys) ? ConvergePhys : NotConverged;

  steps_taken = 0;
  while (steps_taken < max_steps && convergence_status == NotConverged)
  {
    // Store the derivative columns in matrix format.
    Matrix<T,phys_dim,ref_dim> jacobian;
    for (int32 rdim = 0; rdim < ref_dim; rdim++)
    {
      jacobian.set_col(rdim, deriv_cols[rdim]);
    }

    // Compute delta_x by hitting delta_y with the inverse of jacobian.
    bool inverse_valid;
    Vec<T,ref_dim> delta_x;
    delta_x = matrix_mult_inv(jacobian, delta_y, inverse_valid);  //Compiler error if ref_dim != phys_dim.

    if (inverse_valid)
    {
      // Apply the Newton increment.
      x = x + delta_x;
      steps_taken++;

      // If converged, we're done.
      convergence_status = (delta_x.Normlinf() < tol_ref) ? ConvergeRef : NotConverged;
      if (convergence_status == ConvergeRef)
        break;
    }
    else
    {
      // Uh-oh. Some kind of singularity.
      break;
    }

    // Evaluate at current ref pt and measure physical error.
    trans.eval(x, y, deriv_cols);
    delta_y = target - y;
    convergence_status = (delta_y.Normlinf() < tol_phys) ? ConvergePhys : NotConverged;
  }  // end while

  ref = x;
  return convergence_status;
} // newton solve

} // namespace dray

#endif


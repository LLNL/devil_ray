#ifndef DRAY_NEWTON_SOLVER_HPP
#define DRAY_NEWTON_SOLVER_HPP

#include <dray/vec.hpp>
#include <dray/matrix.hpp>
#include <dray/types.hpp>

#include <limits>

namespace dray
{

namespace stats
{
  struct NullIterativeProfile {
    DRAY_EXEC void construct() { }
    DRAY_EXEC void set_num_iter(int32 n_iter) { }
    DRAY_EXEC int32 get_num_iter() { return 0; }
  };
#ifdef DRAY_STATS
  struct IterativeProfile
  {
    int32 m_num_iter;

    DRAY_EXEC void construct() { m_num_iter = 0; }
    DRAY_EXEC void set_num_iter(int32 n_iter) { m_num_iter = n_iter; }
    DRAY_EXEC int32 get_num_iter() { return m_num_iter; }
  };
#else
  using IterativeProfile = NullIterativeProfile;
#endif
} // namespace stats


/**
 * IterativeMethod
 *
 * Must be given a functor that steps (old_coords)-->(new_coords)
 * and returns Continue or Abort.
 *
 * All IterativeMethod does is to keep looping until one of the following criteria is met:
 * - The stepper returns Abort;
 * - The maximum number of steps is reached.
 * - The relative iterative error (in input) falls below a threshold;
 *
 * In either of the first two cases, the method returns false; in the third case, it returns true.
 *
 */
struct IterativeMethod
{
  enum StepStatus { Continue = 0, Abort };
  enum Convergence { NotConverged = 0, Converged };

  static constexpr int32 default_max_steps = 10;
  static constexpr Float default_tol = std::numeric_limits<Float>::epsilon() * 2;

  // User provided stats store.
  template <class StatsT, class VecT, class Stepper>
  DRAY_EXEC
  static Convergence solve(StatsT &iter_prof,
                           Stepper &stepper,
                           VecT &approx_sol,
                           const int32 max_steps = default_max_steps,
                           const Float iter_tol = default_tol)
  {
    int32 steps_taken = 0;
    bool converged = false;
    VecT prev_approx_sol = approx_sol;
    while (steps_taken < max_steps && !converged && stepper(approx_sol) == Continue)
    {
      steps_taken++;
      Float residual = (approx_sol - prev_approx_sol).Normlinf();
      // TODO: just multiply by 2.f?
      //T magnitude = (approx_sol + prev_approx_sol).Normlinf() * 0.5;
      //converged = (residual == 0.0) || (residual / magnitude < iter_tol);
      converged = residual < iter_tol;
      prev_approx_sol = approx_sol;
    }

    iter_prof.set_num_iter(steps_taken);
    return (converged ? Converged : NotConverged);
  }

  // No provided stats store. Uses placeholder stats.
  template <class VecT, class Stepper>
  DRAY_EXEC
  static Convergence solve(Stepper &stepper, VecT &approx_sol,
      const int32 max_steps = default_max_steps, const Float iter_tol = default_tol)
  {
    stats::NullIterativeProfile placeholder_stats;
    return solve(placeholder_stats, stepper, approx_sol, max_steps, iter_tol);
  }
};

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
      const Vec<Float,TransOpType::phys_dim> &target, Vec<Float,TransOpType::ref_dim> &ref,
      const Float tol_phys, const Float tol_ref,
      int32 &steps_taken, const int32 max_steps = 10);

    // A version that also keeps the result of the last evaluation.
  template <class TransOpType>
  DRAY_EXEC static SolveStatus solve(
      TransOpType &trans,
      const Vec<Float,TransOpType::phys_dim> &target, Vec<Float,TransOpType::ref_dim> &ref,
      Vec<Float, TransOpType::phys_dim> &y,
      Vec<Vec<Float, TransOpType::phys_dim>, TransOpType::ref_dim> &deriv_cols,
      const Float tol_phys, const Float tol_ref,
      int32 &steps_taken, const int32 max_steps = 10);
};

template <class TransOpType>
DRAY_EXEC typename NewtonSolve::SolveStatus
NewtonSolve::solve(
    TransOpType &trans,
    const Vec<Float,TransOpType::phys_dim> &target,
    Vec<Float,TransOpType::ref_dim> &ref,
    const Float tol_phys,
    const Float tol_ref,
    int32 &steps_taken,
    const int32 max_steps)
{
  constexpr int32 phys_dim = TransOpType::phys_dim;
  constexpr int32 ref_dim = TransOpType::ref_dim;
  Vec<Float,phys_dim>               y;
  Vec<Vec<Float,phys_dim>,ref_dim>  deriv_cols;

  return solve( trans, target, ref, y, deriv_cols, tol_phys, tol_ref, steps_taken, max_steps);
}


template <class TransOpType>
DRAY_EXEC typename NewtonSolve::SolveStatus
NewtonSolve::solve(
    TransOpType &trans,
    const Vec<Float,TransOpType::phys_dim> &target,
    Vec<Float,TransOpType::ref_dim> &ref,
    Vec<Float, TransOpType::phys_dim> &y,
    Vec<Vec<Float, TransOpType::phys_dim>, TransOpType::ref_dim> &deriv_cols,
    const Float tol_phys,
    const Float tol_ref,
    int32 &steps_taken,
    const int32 max_steps)
{
  // Newton step for IterativeMethod.
  struct Stepper {
    DRAY_EXEC typename IterativeMethod::StepStatus operator()
      (Vec<Float,TransOpType::ref_dim> &x)
    {
      constexpr int32 phys_dim = TransOpType::phys_dim;
      constexpr int32 ref_dim = TransOpType::ref_dim;

      Vec<Float,phys_dim> delta_y;
      Vec<Vec<Float,phys_dim>, ref_dim> j_col;
      m_trans.eval(x, delta_y, j_col);
      delta_y = m_target - delta_y;

      Matrix<Float,phys_dim,ref_dim> jacobian;
      for (int32 rdim = 0; rdim < ref_dim; rdim++)
        jacobian.set_col(rdim, j_col[rdim]);

      bool inverse_valid;
      Vec<Float,ref_dim> delta_x;
      delta_x = matrix_mult_inv(jacobian, delta_y, inverse_valid);  //Compiler error if ref_dim != phys_dim.

      if (!inverse_valid)
        return IterativeMethod::Abort;

      x = x + delta_x;
      return IterativeMethod::Continue;
    }

    TransOpType m_trans;
    Vec<Float,TransOpType::phys_dim> m_target;

  } stepper{ trans, target};

  // Find solution.
  SolveStatus result =
      (IterativeMethod::solve(stepper, ref, max_steps, tol_ref) == IterativeMethod::Converged
      ? ConvergeRef : NotConverged);

  // Evaluate at the solution.
  trans.eval(ref, y, deriv_cols);

  return result;

} // newton solve

} // namespace dray

#endif


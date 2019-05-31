#ifndef DRAY_HIGH_ORDER_INTERSECTION_HPP
#define DRAY_HIGH_ORDER_INTERSECTION_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/newton_solver.hpp>
#include <dray/el_trans.hpp>
#include <dray/vec.hpp>

namespace dray
{

// Everything that is needed for Point in cell is encapsulated in Mesh::world2ref() and MeshElem::eval_inverse().

template <typename T>
struct Intersector_RayIsosurf
{
  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
      const MeshElem<T> &mesh_elem, const FieldElem<T> &field_elem,
      const Vec<T,3> &ray_orig, const Vec<T,3> &ray_dir, T isoval, Vec<T,3> &ref_coords, T &ray_dist, bool use_init_guess = false)
  {
    using IterativeMethod = IterativeMethod<T>;

    // TODO would be nicer as a lambda.

    // Newton step to solve the in-element isosurface intersection problem.
    struct Stepper {
      DRAY_EXEC typename IterativeMethod::StepStatus operator()
        (Vec<T,3+1> &xt) const
      {
        Vec<T,3> &x = *(Vec<T,3> *)&xt[0];
        T &rdist = *(T *)&xt[3];

        // Space jacobian and spatial residual.
        Vec<T,3> delta_y;
        Vec<Vec<T,3>,3> j_col;
        m_transf.eval(x, delta_y, j_col);
        delta_y = m_ray_orig + m_ray_dir * rdist - delta_y;

        // Field gradient and field residual.
        Vec<T,1> _delta_f;                   T &delta_f = _delta_f[0];
        Vec<Vec<T,1>,3> _grad_f;             Vec<T,3> &grad_f = *(Vec<T,3> *)&_grad_f[0];
        m_field.eval(x, _delta_f, _grad_f);
        delta_f = m_isovalue - delta_f;
        
        // Inverse of Jacobian (LU decomposition).
        bool inverse_valid;
        Matrix<T,3,3> jacobian;
        for (int32 rdim = 0; rdim < 3; rdim++)
          jacobian.set_col(rdim, j_col[rdim]);
        MatrixInverse<T,3> jac_inv(jacobian, inverse_valid);

        // Compute adjustments using first-order approximation.
        Vec<T,3> delta_x = jac_inv * delta_y;
        Vec<T,3> delta_x_r = jac_inv * m_ray_dir;

        T delta_r = (delta_f - dot(grad_f, delta_x)) / dot(grad_f, delta_x_r);
        delta_x = delta_x + delta_x_r * delta_r;

        if (!inverse_valid)
          return IterativeMethod::Abort;

        // Apply the step.
        x = x + delta_x;
        rdist = rdist + delta_r;
        return IterativeMethod::Continue;
      }

      Element<T,3,3> m_transf;
      Element<T,3,1> m_field;
      Vec<T,3> m_ray_orig;
      Vec<T,3> m_ray_dir;
      T m_isovalue;
    } stepper{ mesh_elem, field_elem, ray_orig, ray_dir, isoval };

    Vec<T,4> vref_coords{ref_coords[0], ref_coords[1], ref_coords[2], ray_dist};
    if (!use_init_guess)
      for (int32 d = 0; d < 3; d++)   // Always use the ray_dist coordinate, for now.
        vref_coords[d] = 0.5;

    //TODO somewhere else in the program, figure out how to set the precision
    const T tol_ref = 1e-6;
    const int32 max_steps = 10;

    // Find solution.
    bool converged = (IterativeMethod::solve(iter_prof, stepper, vref_coords, max_steps, tol_ref) == IterativeMethod::Converged);

    ref_coords = {vref_coords[0], vref_coords[1], vref_coords[2]};
    ray_dist = vref_coords[3];

    return (converged && mesh_elem.is_inside(ref_coords) && ray_dist > 0);  //TODO use near and far.
  }


  /*
   * Adapters to conform to conform to simpler interfaces.
   */

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof, const MeshAccess<T> &dmesh, const FieldAccess<T> &dfield, int32 el_idx,
      const Vec<T,3> &ray_orig, const Vec<T,3> &ray_dir, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    return intersect(iter_prof, dmesh.get_elem(el_idx), dfield.get_elem(el_idx),
          ray_orig, ray_dir, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(const MeshElem<T> &mesh_elem, const FieldElem<T> &field_elem,
      const Vec<T,3> &ray_orig, const Vec<T,3> &ray_dir, T isoval, Vec<T,3> &ref_coords, T &ray_dist, bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof, mesh_elem, field_elem, ray_orig, ray_dir, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(const MeshAccess<T> &dmesh, const FieldAccess<T> &dfield, int32 el_idx,
      const Vec<T,3> &ray_orig, const Vec<T,3> &ray_dir, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof, dmesh, dfield, el_idx, ray_orig, ray_dir, isoval, ref_coords, ray_dist, use_init_guess);
  }
};


template <typename T>
struct Intersector_RayBoundSurf
{
  // None of this is tested yet. TODO

  static constexpr int32 space_dim = 3;
  using RayType = struct { Vec<T,space_dim> dir; Vec<T,space_dim> orig; };

  static Intersector_RayBoundSurf factory(const ElTransData<T,space_dim> &space_data)
  {
    Intersector_RayBoundSurf i_rbs;
    i_rbs.m_space_ctrl_ptr = space_data.m_ctrl_idx.get_device_ptr_const();
    i_rbs.m_space_val_ptr = space_data.m_values.get_device_ptr_const();
    return i_rbs;
  }

  template <typename TransType>
  DRAY_EXEC
  void operator() (int32 face_idx,        // e.g. for Hex, this is 6*el_id + face_number.
                   TransType trans,       // ElTransRayOp of ElTransOp (space)
                   const RayType &ray_data,      // { ray dir, ray orig }
                   bool &does_intersect,
                   int32 &steps_taken,
                   T &dist,
                   Vec<T,TransType::ref_dim> &ref_pt)
  {
    trans.set_minus_ray_dir(ray_data.dir);
    trans.m_coeff_iter.init_iter(m_space_ctrl_ptr, m_space_val_ptr, trans.p + 1, face_idx);   // Assumes BernsteinBasis as ShapeType.

    const Vec<T, space_dim> &target_pt = ray_data.orig;

    constexpr float32 tol_phys = 0.00001;      // TODO
    constexpr float32 tol_ref  = 0.00001;

    Vec<T, TransType::phys_dim>                           result_y;
    Vec<Vec<T, TransType::phys_dim>, TransType::ref_dim>  result_deriv_cols;  // Unused output argument.

    const typename NewtonSolve<T>::SolveStatus not_converged = NewtonSolve<T>::NotConverged;
    typename NewtonSolve<T>::SolveStatus status = not_converged;
    status = NewtonSolve<T>::solve(trans, target_pt, ref_pt, result_y, result_deriv_cols, tol_phys, tol_ref, steps_taken);
    does_intersect = ( status != not_converged && trans.trans_x.is_inside(ref_pt) );
    dist = (result_y - ray_data.orig).magnitude();
  }

  const int32 *m_space_ctrl_ptr;
  const Vec<T,space_dim> *m_space_val_ptr;
};  // struct Intersector_RayBoundSurf

} // namespace dray


#endif // DRAY_HIGH_ORDER_INTERSECTION_HPP

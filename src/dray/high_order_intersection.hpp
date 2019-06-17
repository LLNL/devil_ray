#ifndef DRAY_HIGH_ORDER_INTERSECTION_HPP
#define DRAY_HIGH_ORDER_INTERSECTION_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/newton_solver.hpp>
#include <dray/el_trans.hpp>
#include <dray/vec.hpp>
#include <dray/ray.hpp>

namespace dray
{

// Everything that is needed for Point in cell is encapsulated in Mesh::world2ref() and MeshElem::eval_inverse().

template <typename T>
struct Intersector_RayIsosurf
{
  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
      const MeshElem<T> &mesh_elem, const FieldElem<T> &field_elem,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist, bool use_init_guess = false)
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
    } stepper{ mesh_elem, field_elem, ray.m_orig, ray.m_dir, isoval };

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

    return (converged && mesh_elem.is_inside(ref_coords) && ray.m_near <= ray_dist && ray_dist < ray.m_far);
  }


  /*
   * Adapters to conform to conform to simpler interfaces.
   */

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof, const MeshAccess<T> &dmesh, const FieldAccess<T> &dfield, int32 el_idx,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    return intersect(iter_prof, dmesh.get_elem(el_idx), dfield.get_elem(el_idx),
          ray, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(const MeshElem<T> &mesh_elem, const FieldElem<T> &field_elem,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist, bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof, mesh_elem, field_elem, ray, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(const MeshAccess<T> &dmesh, const FieldAccess<T> &dfield, int32 el_idx,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof, dmesh, dfield, el_idx, ray, isoval, ref_coords, ray_dist, use_init_guess);
  }
};



template <typename T>
struct Intersector_RayFace
{
  // Returns true if an intersection was found. Otherwise false, and nearest_face_id is not changed.
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
      const MeshElem<T> &mesh_elem, const Ray<T> &ray,
      /// int32 &nearest_face_id,
      Vec<T,3> &ref_coords, T &ray_dist, bool use_init_guess = false)
  {
    int32 num_intersecting_faces = 0;
    for (int32 face_id = 0; face_id < 6; face_id++)
    {
      T trial_ray_dist = ray_dist;
      Vec<T,2> fref_coords;
      FaceElement<T,3> face_elem = mesh_elem.get_face_element(face_id);
      face_elem.ref2fref(ref_coords, fref_coords);

      stats::IterativeProfile face_iter_prof;

      if (intersect(face_iter_prof, face_elem, ray, fref_coords, trial_ray_dist, use_init_guess));
      {
        num_intersecting_faces++;
        if (num_intersecting_faces == 1 || trial_ray_dist < ray_dist)
        {
          /// nearest_face_id = face_id;
          ray_dist = trial_ray_dist;
          face_elem.fref2ref(fref_coords, ref_coords);
        }
      }

      iter_prof.set_num_iter(iter_prof.get_num_iter() + face_iter_prof.get_num_iter());
    }

    return num_intersecting_faces > 0;
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
      const FaceElement<T,3> &face_elem, const Ray<T> &ray,
      Vec<T,2> &fref_coords, T &ray_dist, bool use_init_guess = false)
  {
    using IterativeMethod = IterativeMethod<T>;

    // TODO would be nicer as a lambda.

    // Newton step to solve the element face-ray intersection problem.
    struct Stepper {
      DRAY_EXEC typename IterativeMethod::StepStatus operator()
        (Vec<T,2+1> &xt) const
      {
        Vec<T,2> &x = *(Vec<T,2> *)&xt[0];
        T &rdist = *(T *)&xt[2];

        // Space jacobian and spatial residual.
        Vec<T,3> delta_y;
        Vec<Vec<T,3>,2> j_col;
        m_transf.eval(x, delta_y, j_col);
        delta_y = m_ray_orig - delta_y;
        
        Matrix<T,3,3> jacobian;
        jacobian.set_col(0, j_col[0]);
        jacobian.set_col(1, j_col[1]);
        jacobian.set_col(2, -m_ray_dir);

        // Inverse of Jacobian (LU decomposition) --> spatial adjustment and new ray dist.
        bool inverse_valid;
        Vec<T,3> delta_xt = MatrixInverse<T,3>(jacobian, inverse_valid) * delta_y;

        if (!inverse_valid)
          return IterativeMethod::Abort;

        // Apply the step.
        x = x + (*(Vec<T,2> *)&delta_xt);
        rdist = delta_xt[2];
        return IterativeMethod::Continue;
      }

      FaceElement<T,3> m_transf;
      Vec<T,3> m_ray_orig;
      Vec<T,3> m_ray_dir;
    } stepper{ face_elem, ray.m_orig, ray.m_dir };

    Vec<T,2+1> vref_coords{fref_coords[0], fref_coords[1], ray_dist};
    if (!use_init_guess)
      for (int32 d = 0; d < 2; d++)   // Always use the ray_dist coordinate, for now.
        vref_coords[d] = 0.5;

    //TODO somewhere else in the program, figure out how to set the precision
    const T tol_ref = 1e-6;
    const int32 max_steps = 10;

    // Find solution.
    bool converged = (IterativeMethod::solve(iter_prof, stepper, vref_coords, max_steps, tol_ref) == IterativeMethod::Converged);

    fref_coords = {vref_coords[0], vref_coords[1]};
    ray_dist = vref_coords[2];

    return (converged && face_elem.is_inside(fref_coords) && ray.m_near <= ray_dist && ray_dist < ray.m_far);
  }

  /* TODO adapters */
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
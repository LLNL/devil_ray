#ifndef DRAY_HIGH_ORDER_INTERSECTION_HPP
#define DRAY_HIGH_ORDER_INTERSECTION_HPP

#include <dray/newton_solver.hpp>
#include <dray/el_trans.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <typename T>
struct Intersector_PointVol
{
  static constexpr int32 phys_dim = 3;
  using RayType = Vec<T,phys_dim>;

  static Intersector_PointVol factory(const ElTransData<T,phys_dim> &space_data)
  {
    Intersector_PointVol i_pv;
    i_pv.m_ctrl_idx_ptr = space_data.m_ctrl_idx.get_device_ptr_const();
    i_pv.m_values_ptr = space_data.m_values.get_device_ptr_const();
    return i_pv;
  }

    // Assumes that the aux_mem for trans has already been set up.
    // Modifies trans.
  template <typename TransType>
  DRAY_EXEC
  void operator() (int32 el_idx,
                   TransType trans,             // ElTransOp of space.
                   const RayType &ray_data,      // This is just a Vec3 in the case of PointVol.
                   bool &does_intersect,
                   int32 &steps_taken,
                   T &dist,
                   Vec<T,TransType::ref_dim> &ref_pt)
  {
    trans.m_coeff_iter.init_iter(m_ctrl_idx_ptr, m_values_ptr, trans.get_el_dofs(), el_idx);
    const Vec<T,phys_dim> &target_pt = ray_data;

    constexpr float32 tol_phys = 0.00001;      // TODO
    constexpr float32 tol_ref  = 0.00001;

    const typename NewtonSolve<T>::SolveStatus not_converged = NewtonSolve<T>::NotConverged;
    typename NewtonSolve<T>::SolveStatus status = not_converged;
    status = NewtonSolve<T>::solve(trans, target_pt, ref_pt, tol_phys, tol_ref, steps_taken);
    does_intersect = ( status != not_converged && TransType::is_inside(ref_pt) );
    // dist is not modified.
  }

  const int32 *m_ctrl_idx_ptr;
  const Vec<T,phys_dim> *m_values_ptr;
};  // struct Intersector_PointVol


template <typename T>
struct Intersector_RayIsosurf
{
  static constexpr int32 space_dim = 3;
  static constexpr int32 field_dim = 1;
  using RayType = struct { Vec<T,space_dim> dir; Vec<T,space_dim> orig; T isoval; };

  static Intersector_RayIsosurf factory(const ElTransData<T,space_dim> &space_data, const ElTransData<T,field_dim> &field_data)
  {
    Intersector_RayIsosurf i_ri;
    i_ri.m_space_ctrl_ptr = space_data.m_ctrl_idx.get_device_ptr_const();
    i_ri.m_field_ctrl_ptr = field_data.m_ctrl_idx.get_device_ptr_const();
    i_ri.m_space_val_ptr = space_data.m_values.get_device_ptr_const();
    i_ri.m_field_val_ptr = field_data.m_values.get_device_ptr_const();
    return i_ri;
  }

    // Assumes that the aux_mem for trans has already been set up.
  template <typename TransType>
  DRAY_EXEC
  void operator() (int32 el_idx,
                   TransType trans,       // ElTransRayOp of ElTransPairOp (space + field)
                   const RayType &ray_data,      // { ray dir, ray orig, isoval }
                   bool &does_intersect,
                   int32 &steps_taken,
                   T &dist,
                   Vec<T,TransType::ref_dim> &ref_pt)
  {
    trans.set_minus_ray_dir(ray_data.dir);
    trans.trans_x.m_coeff_iter.init_iter(m_space_ctrl_ptr, m_space_val_ptr, trans.trans_x.get_el_dofs(), el_idx);
    trans.trans_y.m_coeff_iter.init_iter(m_field_ctrl_ptr, m_field_val_ptr, trans.trans_y.get_el_dofs(), el_idx);

    const Vec<T, space_dim + field_dim> &target_pt = (const Vec<T, space_dim + field_dim> &) ray_data.orig; // includes isoval.

    constexpr float32 tol_phys = 0.00001;      // TODO
    constexpr float32 tol_ref  = 0.00001;

    Vec<T, TransType::phys_dim>                           result_y;
    Vec<Vec<T, TransType::phys_dim>, TransType::ref_dim>  result_deriv_cols;  // Unused output argument.

    const typename NewtonSolve<T>::SolveStatus not_converged = NewtonSolve<T>::NotConverged;
    typename NewtonSolve<T>::SolveStatus status = not_converged;
    status = NewtonSolve<T>::solve(trans, target_pt, ref_pt, result_y, result_deriv_cols, tol_phys, tol_ref, steps_taken);
    does_intersect = ( status != not_converged && trans.is_inside(ref_pt));
    dist = ((Vec<T,space_dim> &) result_y - ray_data.orig).magnitude();
  }

  const int32 *m_space_ctrl_ptr;
  const int32 *m_field_ctrl_ptr;
  const Vec<T,space_dim> *m_space_val_ptr;
  const Vec<T,field_dim> *m_field_val_ptr;
};  // struct Intersector_RayIsosurf


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

    // Assumes that the aux_mem for trans has already been set up.
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

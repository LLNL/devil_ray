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
    //TODO once we have IterativeMethod, it would be fine to implement a NewtonStep right here.
    // for now, put together a transformation object and call NewtonSolve<T>::solve().

    using RayElemIsovalueTransform = ElTransRayOp<T, ElTransPairOp<T, MeshElem<T>, FieldElem<T>>, 3>;
    RayElemIsovalueTransform vtrans;
    vtrans.trans_x = mesh_elem;
    vtrans.trans_y = field_elem;
    vtrans.set_minus_ray_dir(ray_dir);

    const Vec<T,4> vtarget{ray_orig[0], ray_orig[1], ray_orig[2], isoval};
    Vec<T,4> vref_coords{ref_coords[0], ref_coords[1], ref_coords[2], ray_dist};

    const T tol_phys = 0.00001;
    const T tol_ref = 0.00001;
    const int32 max_steps = 10;

    if (!use_init_guess)
      for (int32 d = 0; d < 3; d++)   // Always use the ray_dist coordinate, for now.
        vref_coords[d] = 0.5;

    int32 iterative_counter = 0;                                       //TODO pass to a persistent counter.
    typename NewtonSolve<T>::SolveStatus result =
        NewtonSolve<T>::solve( vtrans, vtarget, vref_coords,
            tol_phys, tol_ref, iterative_counter, max_steps );

#ifdef DRAY_STATS
    iter_prof.m_num_iter = iterative_counter;
#endif

    ref_coords = {vref_coords[0], vref_coords[1], vref_coords[2]};
    ray_dist = vref_coords[3];

    return (result != NewtonSolve<T>::NotConverged && RayElemIsovalueTransform::is_inside(vref_coords));
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

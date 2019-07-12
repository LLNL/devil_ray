#ifndef DRAY_HIGH_ORDER_INTERSECTION_HPP
#define DRAY_HIGH_ORDER_INTERSECTION_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/newton_solver.hpp>
#include <dray/el_trans.hpp>
#include <dray/vec.hpp>
#include <dray/ray.hpp>
#include <dray/utils/ray_utils.hpp>

namespace dray
{

// Everything that is needed for Point in cell is encapsulated in Mesh::world2ref() and MeshElem::eval_inverse().

template <typename T>
struct Intersector_RayIsosurf
{
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
                                  const MeshElem<T> &mesh_elem,
                                  const FieldElem<T> &field_elem,
                                  const Ray<T> &ray,
                                  T isoval,
                                  const AABB<3> &guess_domain,
                                  Vec<T,3> &ref_coords,
                                  T & ray_dist,
                                  bool use_init_guess = false)
  {
    using StateT = stats::IterativeProfile;
    using QueryT = std::pair<Ray<T>, T>;
    using ElemT = std::pair<MeshElem<T>, FieldElem<T>>;
    using RefBoxT = AABB<3>;
    using SolT = Vec<T,4>;

    //TODO need a few more subdivisions to fix holes
    const T tol_refbox = 1e-2;
    constexpr int32 subdiv_budget = 0;

    RefBoxT domain = (use_init_guess ? guess_domain : AABB<3>::ref_universe());

    const QueryT ray_iso_query{ray, isoval};
    const ElemT element{mesh_elem, field_elem};

    // For subdivision search, test whether the sub-element possibly contains the query point.
    // Strict test because the bounding boxes are approximate.
    struct FInBounds { DRAY_EXEC bool operator()(StateT &state, const QueryT &query, const ElemT &elem, const RefBoxT &ref_box) {
      dray::AABB<3> space_bounds;
      dray::AABB<1> field_bounds;
      elem.first.get_sub_bounds(ref_box.m_ranges, space_bounds);
      elem.second.get_sub_bounds(ref_box.m_ranges, field_bounds);

      // Test isovalue \in isorange.
      const T &isovalue = query.second;
      bool in_bounds_field = field_bounds.m_ranges[0].min() <= isovalue && isovalue < field_bounds.m_ranges[0].max();  // Test isovalue \in isorange.

      // Test intersection of bounding box with ray.
      const Ray<T> &qray = query.first;
      bool ray_bounds = intersect_ray_aabb(qray, space_bounds);

      return in_bounds_field && ray_bounds;
    } };

    // Get solution when close enough: Iterate using Newton's method.
    struct FGetSolution { DRAY_EXEC bool operator()(StateT &state, const QueryT &query, const ElemT &elem, const RefBoxT &ref_box, SolT &solution) {
      Vec<T,3> sol_ref_coords = ref_box.template center<T>();
      T sol_ray_dist = query.first.m_near;
      stats::IterativeProfile &iterations = state;

      bool sol_found = intersect_local(iterations,
                                       elem.first,
                                       elem.second,
                                       query.first,
                                       query.second,
                                       sol_ref_coords,
                                       sol_ray_dist,
                                       true);
      solution[0] = sol_ref_coords[0];                         // Pack
      solution[1] = sol_ref_coords[1];
      solution[2] = sol_ref_coords[2];
      solution[3] = sol_ray_dist;

      return sol_found;
    } };

    // Initiate subdivision search.
    SolT solution;
    uint32 ret_code;
    int32 num_solutions = SubdivisionSearch::subdivision_search
        <StateT, QueryT, ElemT, T, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget>(
        ret_code, iter_prof, ray_iso_query, element, tol_refbox, &domain, &solution, 1);

    ref_coords[0] = solution[0];                               // Unpack
    ref_coords[1] = solution[1];
    ref_coords[2] = solution[2];
    ray_dist = solution[3];

    return num_solutions > 0;
  }


  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(stats::IterativeProfile &iter_prof,
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
    const T tol_ref = 1e-4;
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
  DRAY_EXEC static bool intersect(
      stats::IterativeProfile &iter_prof,
      const MeshAccess<T> &dmesh,
      const FieldAccess<T> &dfield,
      int32 el_idx,
      const Ray<T> &ray,
      T isoval,
      const AABB<3> &guess_domain,
      Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    return intersect(iter_prof,
                     dmesh.get_elem(el_idx),
                     dfield.get_elem(el_idx),
                     ray,
                     isoval,
                     guess_domain,
                     ref_coords,
                     ray_dist,
                     use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(
      const MeshElem<T> &mesh_elem, const FieldElem<T> &field_elem,
      const Ray<T> &ray,
      T isoval,
      const AABB<3> &guess_domain,
      Vec<T,3> &ref_coords,
      T &ray_dist,
      bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof,
                     mesh_elem,
                     field_elem,
                     ray,
                     isoval,
                     guess_domain,
                     ref_coords,
                     ray_dist,
                     use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect(
      const MeshAccess<T> &dmesh,
      const FieldAccess<T> &dfield,
      int32 el_idx,
      const Ray<T> &ray,
      T isoval,
      const AABB<3> &guess_domain,
      Vec<T,3> &ref_coords,
      T &ray_dist,
      bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof,
                     dmesh,
                     dfield,
                     el_idx,
                     ray,
                     isoval,
                     guess_domain,
                     ref_coords,
                     ray_dist,
                     use_init_guess);
  }



  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(stats::IterativeProfile &iter_prof, const MeshAccess<T> &dmesh, const FieldAccess<T> &dfield, int32 el_idx,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    return intersect_local(iter_prof, dmesh.get_elem(el_idx), dfield.get_elem(el_idx),
          ray, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(const MeshElem<T> &mesh_elem, const FieldElem<T> &field_elem,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist, bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect_local(iter_prof, mesh_elem, field_elem, ray, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(const MeshAccess<T> &dmesh, const FieldAccess<T> &dfield, int32 el_idx,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect_local(iter_prof, dmesh, dfield, el_idx, ray, isoval, ref_coords, ray_dist, use_init_guess);
  }
};



template <typename T>
struct Intersector_RayFace
{

  //
  // intersect() (all faces of an element)
  //
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
                                  const MeshElem<T> &mesh_elem,
                                  const Ray<T> &ray,
                                  const AABB<3> &guess_domain,
                                  Vec<T,3> &ref_coords,
                                  T &ray_dist,
                                  bool use_init_guess = false)   // TODO add ref box guess
  {
    int32 num_intersecting_faces = 0;
    for (int32 face_id = 0; face_id < 6; face_id++)
    {
      T trial_ray_dist = ray_dist;

      FaceElement<T,3> face_elem = mesh_elem.get_face_element(face_id);

      Vec<T,2> fref_coords;
      face_elem.ref2fref(ref_coords, fref_coords);

      AABB<2> face_guess_domain;
      bool projection_nonempty;
      face_elem.ref2fref(guess_domain, face_guess_domain, projection_nonempty);

      stats::IterativeProfile face_iter_prof;

      if (projection_nonempty && intersect(face_iter_prof,
                                           face_elem,
                                           ray,
                                           face_guess_domain,
                                           fref_coords,
                                           trial_ray_dist,
                                           use_init_guess))
      {
        num_intersecting_faces++;
        if (num_intersecting_faces == 1 || trial_ray_dist < ray_dist)
        {
          /// nearest_face_id = face_id;
          ray_dist = trial_ray_dist;
          face_elem.fref2ref(fref_coords, ref_coords);
          face_elem.set_face_coordinate(ref_coords);
        }
      }

      iter_prof.set_num_iter(iter_prof.get_num_iter() + face_iter_prof.get_num_iter());
    }

    return num_intersecting_faces > 0;
  }


  //
  // intersect() (single face only)
  //
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
                                  const FaceElement<T,3> &face_elem,
                                  const Ray<T> &ray,
                                  const AABB<2> &face_guess_domain,
                                  Vec<T,2> &fref_coords,
                                  T &ray_dist,
                                  bool use_init_guess = false)
  {
    using StateT = std::pair<stats::IterativeProfile, int32>;  //(iterations, DEBUG pixel_id)
    using QueryT = Ray<T>;
    using ElemT = FaceElement<T,3>;
    using RefBoxT = AABB<2>;
    using SolT = Vec<T,3>;

    const T tol_refbox = 1e-2;
    constexpr int32 subdiv_budget = 100;   // 0 means initial_guess = face_guess_domain.center();

    RefBoxT domain = (use_init_guess ? face_guess_domain : AABB<2>::ref_universe());

    const QueryT &ray_query = ray;

    // For subdivision search, test whether the sub-element possibly contains the query point.
    // Strict test because the bounding boxes are approximate.
    struct FInBounds { DRAY_EXEC bool operator()(StateT &state, const QueryT &query, const ElemT &elem, const RefBoxT &ref_box) {
      // Test intersection of bounding box with ray.
      dray::AABB<3> space_bounds;
      elem.get_sub_bounds(ref_box.m_ranges, space_bounds);
      bool ray_bounds = intersect_ray_aabb(query, space_bounds);
      return ray_bounds;
    } };

    // Get solution when close enough: Iterate using Newton's method.
    struct FGetSolution { DRAY_EXEC bool operator()(StateT &state, const QueryT &query, const ElemT &elem, const RefBoxT &ref_box, SolT &solution) {

      Vec<T,2> sol_fref_coords = ref_box.template center<T>();
      T sol_ray_dist = query.m_near;

      stats::IterativeProfile &iterations = state.first;
      bool sol_found = intersect_local(iterations, elem, query, sol_fref_coords, sol_ray_dist, true);

      solution[0] = sol_fref_coords[0];                         // Pack
      solution[1] = sol_fref_coords[1];
      solution[2] = sol_ray_dist;

      return sol_found;
    } };

    // Initiate subdivision search.
    //TODO there could be several solutions... restructure surface intersection to allow for several.
    SolT solution;
    uint32 ret_code;
    StateT state_ob{iter_prof, ray.m_pixel_id};
    int32 num_solutions = SubdivisionSearch::subdivision_search
        <StateT, QueryT, ElemT, T, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget>(
        ret_code, state_ob, ray_query, face_elem, tol_refbox, &domain, &solution, 1);

    fref_coords[0] = solution[0];                             // Unpack.
    fref_coords[1] = solution[1];
    ray_dist = solution[2];

    return num_solutions > 0;
  }


  //
  // intersect_local() (all faces of an element)
  //
  DRAY_EXEC static bool intersect_local(stats::IterativeProfile &iter_prof,
                                        const MeshElem<T> &mesh_elem,
                                        const Ray<T> &ray,
                                        Vec<T,3> &ref_coords,
                                        T &ray_dist,
                                        bool use_init_guess = false)
  {
    int32 num_intersecting_faces = 0;
    for (int32 face_id = 0; face_id < 6; face_id++)
    {
      T trial_ray_dist = ray_dist;
      Vec<T,2> fref_coords;
      FaceElement<T,3> face_elem = mesh_elem.get_face_element(face_id);
      face_elem.ref2fref(ref_coords, fref_coords);

      stats::IterativeProfile face_iter_prof;

      if (intersect_local(face_iter_prof, face_elem, ray, fref_coords, trial_ray_dist, use_init_guess))
      {
        num_intersecting_faces++;
        if (num_intersecting_faces == 1 || trial_ray_dist < ray_dist)
        {
          /// nearest_face_id = face_id;
          ray_dist = trial_ray_dist;
          face_elem.fref2ref(fref_coords, ref_coords);
          face_elem.set_face_coordinate(ref_coords);
        }
      }

      iter_prof.set_num_iter(iter_prof.get_num_iter() + face_iter_prof.get_num_iter());
    }

    return num_intersecting_faces > 0;
  }



  //
  // intersect_local() (single face only)
  //
  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(stats::IterativeProfile &iter_prof,
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
    const T tol_ref = 1e-4;
    const int32 max_steps = 10;

    // Find solution.
    bool converged = (IterativeMethod::solve(iter_prof, stepper, vref_coords, max_steps, tol_ref) == IterativeMethod::Converged);

    fref_coords = {vref_coords[0], vref_coords[1]};
    ray_dist = vref_coords[2];

    return (converged && face_elem.is_inside(fref_coords) && ray.m_near <= ray_dist && ray_dist < ray.m_far);
  }

  /* TODO adapters */
};


} // namespace dray


#endif // DRAY_HIGH_ORDER_INTERSECTION_HPP

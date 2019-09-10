#ifndef DRAY_HIGH_ORDER_INTERSECTION_HPP
#define DRAY_HIGH_ORDER_INTERSECTION_HPP

/**
 * @file high_order_intersection.hpp
 * @brief (Relatively) High-level interface for kernels of filters
 *        to do high-order ray intersections.
 *
 * There are two types of ray intersections:
 * - (Ray)  x  (portion of isosurface in a volumetric scalar field element);
 * - (Ray)  x  (face in a high order surface mesh).
 */

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/newton_solver.hpp>
#include <dray/vec.hpp>
#include <dray/ray.hpp>
#include <dray/utils/ray_utils.hpp>

namespace dray
{

// Everything that is needed for Point in cell is encapsulated in Mesh::world2ref() and MeshElem::eval_inverse().

  // This is only supported for 3D elements and 1D fields.
template <typename T, class ElemT>
struct Intersector_RayIsosurf
{
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
                                  const ElemT &mesh_elem,
                                  const FieldOn<ElemT, 1u> &field_elem,
                                  const Ray<T> &ray,
                                  T isoval,
                                  const AABB<3> &guess_domain,
                                  Vec<T,3> &ref_coords,
                                  T & ray_dist,
                                  bool use_init_guess = false)
  {
    using StateT = stats::IterativeProfile;
    using QueryT = std::pair<Ray<T>, T>;
    using ElemPair = std::pair<ElemT, FieldOn<ElemT, 1u>>;
    using RefBoxT = AABB<3>;
    using SolT = Vec<T,4>;

    //TODO need a few more subdivisions to fix holes
    const T tol_refbox = 1e-2;
    constexpr int32 subdiv_budget = 0;

    RefBoxT domain = (use_init_guess ? guess_domain : AABB<3>::ref_universe());

    const QueryT ray_iso_query{ray, isoval};
    const ElemPair element{mesh_elem, field_elem};

    // For subdivision search, test whether the sub-element possibly contains the query point.
    // Strict test because the bounding boxes are approximate.
    struct FInBounds
    {
      DRAY_EXEC bool operator()(StateT &state,
                                const QueryT &query,
                                const ElemPair &elem,
                                const RefBoxT &ref_box)
      {
        dray::AABB<3> space_bounds;
        dray::AABB<1> field_bounds;
        elem.first.get_sub_bounds(ref_box, space_bounds);
        elem.second.get_sub_bounds(ref_box, field_bounds);

        // Test isovalue \in isorange.
        const T &isovalue = query.second;
        // Test isovalue \in isorange.
        bool in_bounds_field = field_bounds.m_ranges[0].min() <= isovalue && isovalue < field_bounds.m_ranges[0].max();

        // Test intersection of bounding box with ray.
        const Ray<T> &qray = query.first;
        bool ray_bounds = intersect_ray_aabb(qray, space_bounds);

        return in_bounds_field && ray_bounds;
    } };

    // Get solution when close enough: Iterate using Newton's method.
    struct FGetSolution
    {
      DRAY_EXEC bool operator()(StateT &state,
                                const QueryT &query,
                                const ElemPair &elem,
                                const RefBoxT &ref_box,
                                SolT &solution)
      {
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
      }
    };

    // Initiate subdivision search.
    SolT solution;
    uint32 ret_code;
    int32 num_solutions = SubdivisionSearch::subdivision_search
        <StateT, QueryT, ElemPair, T, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget>(
        ret_code, iter_prof, ray_iso_query, element, tol_refbox, &domain, &solution, 1);

    ref_coords[0] = solution[0];                               // Unpack
    ref_coords[1] = solution[1];
    ref_coords[2] = solution[2];
    ray_dist = solution[3];

    return num_solutions > 0;
  }


  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(stats::IterativeProfile &iter_prof,
      const ElemT &mesh_elem, const FieldOn<ElemT, 1u> &field_elem,
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
        delta_y = m_transf.eval_d(x, j_col);
        delta_y = m_ray_orig + m_ray_dir * rdist - delta_y;

        // Field gradient and field residual.
        Vec<T,1> _delta_f;                   T &delta_f = _delta_f[0];
        Vec<Vec<T,1>,3> _grad_f;             Vec<T,3> &grad_f = *(Vec<T,3> *)&_grad_f[0];
        _delta_f = m_field.eval_d(x, _grad_f);
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

      ElemT m_transf;
      FieldOn<ElemT, 1u> m_field;
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
      const MeshAccess<T, ElemT> &dmesh,
      const FieldAccess<T, FieldOn<ElemT, 1u>> &dfield,
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
      const ElemT &mesh_elem, const FieldOn<ElemT, 1u> &field_elem,
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
      const MeshAccess<T, ElemT> &dmesh,
      const FieldAccess<T, FieldOn<ElemT, 1u>> &dfield,
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
  DRAY_EXEC static bool intersect_local(stats::IterativeProfile &iter_prof, const MeshAccess<T, ElemT> &dmesh, const FieldAccess<T, FieldOn<ElemT, 1u>> &dfield, int32 el_idx,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    return intersect_local(iter_prof, dmesh.get_elem(el_idx), dfield.get_elem(el_idx),
          ray, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(const ElemT &mesh_elem, const FieldOn<ElemT, 1u> &field_elem,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist, bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect_local(iter_prof, mesh_elem, field_elem, ray, isoval, ref_coords, ray_dist, use_init_guess);
  }

  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(const MeshAccess<T, ElemT> &dmesh, const FieldAccess<T, FieldOn<ElemT, 1u>> &dfield, int32 el_idx,
      const Ray<T> &ray, T isoval, Vec<T,3> &ref_coords, T &ray_dist,
      bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect_local(iter_prof, dmesh, dfield, el_idx, ray, isoval, ref_coords, ray_dist, use_init_guess);
  }
};





// The new intersector for elements of dimension 2.
template <typename T, class ElemT>
struct Intersector_RayFace
{
  DRAY_EXEC static bool intersect(stats::IterativeProfile &iter_prof,
                                  const ElemT &surf_elem,
                                  const Ray<T> &ray,
                                  const AABB<2> &guess_domain,
                                  Vec<T,2> &ref_coords,
                                  T & ray_dist,
                                  bool use_init_guess = false)
  {
    using StateT = stats::IterativeProfile;
    using QueryT = Ray<T>;
    using RefBoxT = AABB<2>;
    using SolT = Vec<T,3>;    // Parametric coordinates + ray distance.

    // TODO?
    const T tol_refbox = 1e-2;
    constexpr int32 subdiv_budget = 0;

    RefBoxT domain = (use_init_guess ? guess_domain : RefBoxT::ref_universe());

    // For subdivision search, test whether the sub-element possibly contains the query point.
    // Strict test because the bounding boxes are approximate.
    struct FInBounds
    {
      DRAY_EXEC bool operator()(StateT &state,
                                const QueryT &qray,
                                const ElemT &elem,
                                const RefBoxT &ref_box)
      {
        // Test intersection of bounding box with ray.
        dray::AABB<3> space_bounds;
        elem.get_sub_bounds(ref_box, space_bounds);
        return intersect_ray_aabb(qray, space_bounds);
    } };

    // Get solution when close enough: Iterate using Newton's method.
    struct FGetSolution
    {
      DRAY_EXEC bool operator()(StateT &state,
                                const QueryT &qray,
                                const ElemT &elem,
                                const RefBoxT &ref_box,
                                SolT &solution)
      {
        Vec<T,2> sol_ref_coords = ref_box.template center<T>();
        T sol_ray_dist = qray.m_near;
        stats::IterativeProfile &iterations = state;

        bool sol_found = Intersector_RayFace<T,ElemT>::intersect_local(iterations,
                                                                       elem,
                                                                       qray,
                                                                       sol_ref_coords,
                                                                       sol_ray_dist,
                                                                       true);
        solution[0] = sol_ref_coords[0];         // Pack
        solution[1] = sol_ref_coords[1];
        solution[2] = sol_ray_dist;

        return sol_found;
      }
    };

    // Initiate subdivision search.
    SolT solution;
    uint32 ret_code;
    int32 num_solutions = SubdivisionSearch::subdivision_search
        <StateT, QueryT, ElemT, T, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget>(
        ret_code, iter_prof, ray, surf_elem, tol_refbox, &domain, &solution, 1);

    ref_coords[0] = solution[0];                 // Unpack
    ref_coords[1] = solution[1];
    ray_dist = solution[2];

    return num_solutions > 0;
  }


  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local(stats::IterativeProfile &iter_prof,
                                        const ElemT &surf_elem,
                                        const Ray<T> &ray,
                                        Vec<T,2> &ref_coords,
                                        T &ray_dist,
                                        bool use_init_guess = false)
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
        delta_y = m_transf.eval_d(x, j_col);
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

      ElemT m_transf;
      Vec<T,3> m_ray_orig;
      Vec<T,3> m_ray_dir;
    } stepper{ surf_elem, ray.m_orig, ray.m_dir };

    Vec<T,2+1> vref_coords{ref_coords[0], ref_coords[1], ray_dist};
    if (!use_init_guess)
      for (int32 d = 0; d < 2; d++)   // Always use the ray_dist coordinate, for now.
        vref_coords[d] = 0.5;

    //TODO somewhere else in the program, figure out how to set the precision
    const T tol_ref = 1e-4;
    const int32 max_steps = 10;

    // Find solution.
    bool converged = (IterativeMethod::solve(iter_prof,
                                             stepper,
                                             vref_coords,
                                             max_steps,
                                             tol_ref) == IterativeMethod::Converged);

    ref_coords = {vref_coords[0], vref_coords[1]};
    ray_dist = vref_coords[2];

    return (converged && surf_elem.is_inside(ref_coords) && ray.m_near <= ray_dist && ray_dist < ray.m_far);
  }

  /*
   * Adapters to conform to simpler interfaces.
   */

  // Returns true if an intersection was found.
  // Adapter: Substitutes element (dmesh.get_elem(el_idx)).
  DRAY_EXEC static bool intersect( stats::IterativeProfile &iter_prof,
                                   const MeshAccess<T, ElemT> &dmesh,  // Should be 2D device mesh.
                                   int32 el_idx,
                                   const Ray<T> &ray,
                                   const AABB<2> &guess_domain,
                                   Vec<T,2> &ref_coords,
                                   T &ray_dist,
                                   bool use_init_guess = false)
  {
    return intersect(iter_prof,
                     dmesh.get_elem(el_idx),  // <--- substitute element.
                     ray,
                     guess_domain,
                     ref_coords,
                     ray_dist,
                     use_init_guess);
  }

  // Returns true if an intersection was found.
  // Adapter: Substitutes dummy iter_prof (iteration counter).
  DRAY_EXEC static bool intersect( const ElemT &mesh_elem,
                                   const Ray<T> &ray,
                                   const AABB<2> &guess_domain,
                                   Vec<T,2> &ref_coords,
                                   T &ray_dist,
                                   bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof,          // <--- substitute dummy iter_prof.
                     mesh_elem,
                     ray,
                     guess_domain,
                     ref_coords,
                     ray_dist,
                     use_init_guess);
  }

  // Returns true if an intersection was found.
  // Adapter: Both substitutes mesh element and provides dummy iter_prof.
  DRAY_EXEC static bool intersect( const MeshAccess<T, ElemT> &dmesh,
                                   int32 el_idx,
                                   const Ray<T> &ray,
                                   const AABB<2> &guess_domain,
                                   Vec<T,2> &ref_coords,
                                   T &ray_dist,
                                   bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect(iter_prof,          // <--- substitute dummy iter_prof.
                     dmesh,              // <--- (dmesh, el_idx) will be substituted.
                     el_idx,             //
                     ray,
                     guess_domain,
                     ref_coords,
                     ray_dist,
                     use_init_guess);
  }



  // Returns true if an intersection was found.
  // Adapter: Substitutes element (dmesh.get_elem(el_idx)).
  DRAY_EXEC static bool intersect_local( stats::IterativeProfile &iter_prof,
                                         const MeshAccess<T, ElemT> &dmesh,
                                         int32 el_idx,
                                         const Ray<T> &ray,
                                         Vec<T,2> &ref_coords,
                                         T &ray_dist,
                                         bool use_init_guess = false)
  {
    return intersect_local(iter_prof,
                           dmesh.get_elem(el_idx),   // <--- substitute mesh element.
                           ray,
                           ref_coords,
                           ray_dist,
                           use_init_guess);
  }

  // Returns true if an intersection was found.
  // Adapter: Substitutes dummy iter_prof (iteration counter).
  DRAY_EXEC static bool intersect_local( const ElemT &mesh_elem,
                                         const Ray<T> &ray,
                                         Vec<T,2> &ref_coords,
                                         T &ray_dist,
                                         bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect_local(iter_prof,       // <--- substitute dummy iter_prof.
                           mesh_elem,
                           ray,
                           ref_coords,
                           ray_dist,
                           use_init_guess);
  }

  // Returns true if an intersection was found.
  // Adapter: Both substitutes mesh element and provides dummy iter_prof.
  DRAY_EXEC static bool intersect_local( const MeshAccess<T, ElemT> &dmesh,
                                         int32 el_idx,
                                         const Ray<T> &ray,
                                         Vec<T,2> &ref_coords,
                                         T &ray_dist,
                                         bool use_init_guess = false)
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return intersect_local(iter_prof,     // <--- substitute dummy iter_prof.
                           dmesh,         // <--- (dmesh, el_idx) will be substituted.
                           el_idx,        //
                           ray,
                           ref_coords,
                           ray_dist,
                           use_init_guess);
  }
};



} // namespace dray


#endif // DRAY_HIGH_ORDER_INTERSECTION_HPP

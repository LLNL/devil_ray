#ifndef DRAY_ISOSURFACE_INTERSECTION_HPP
#define DRAY_ISOSURFACE_INTERSECTION_HPP

#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/newton_solver.hpp>
#include <dray/ray.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/vec.hpp>

namespace dray
{

// TODO i don't think this should be a class

template <class ElemT> struct Intersector_RayIsosurf
{
  DRAY_EXEC static bool intersect (stats::Stats &stats,
                                   const ElemT &mesh_elem,
                                   const FieldOn<ElemT, 1u> &field_elem,
                                   const Ray &ray,
                                   Float isoval,
                                   const AABB<3> &guess_domain,
                                   Vec<Float, 3> &ref_coords,
                                   Float &ray_dist,
                                   bool use_init_guess = false)
  {
    using QueryT = std::pair<Ray, Float>;
    using ElemPair = std::pair<ElemT, FieldOn<ElemT, 1u>>;
    using RefBoxT = AABB<3>;
    using SolT = Vec<Float, 4>;

    // TODO need a few more subdivisions to fix holes
    const Float tol_refbox = 1e-2;
    constexpr int32 subdiv_budget = 0;

    RefBoxT domain = (use_init_guess ? guess_domain : AABB<3>::ref_universe ());

    const QueryT ray_iso_query{ ray, isoval };
    const ElemPair element{ mesh_elem, field_elem };

    // For subdivision search, test whether the sub-element
    // possibly contains the query point.
    // Strict test because the bounding boxes are approximate.
    struct FInBounds
    {
      DRAY_EXEC bool operator() (stats::Stats &stats,
                                 const QueryT &query,
                                 const ElemPair &elem,
                                 const RefBoxT &ref_box)
      {
        dray::AABB<3> space_bounds;
        dray::AABB<1> field_bounds;
        elem.first.get_sub_bounds (ref_box, space_bounds);
        elem.second.get_sub_bounds (ref_box, field_bounds);

        // Test isovalue \in isorange.
        const Float &isovalue = query.second;
        // Test isovalue \in isorange.
        bool in_bounds_field = false;

        if (field_bounds.m_ranges[0].min () <= isovalue &&
            isovalue < field_bounds.m_ranges[0].max ())
        {
          in_bounds_field = true;
        }

        // Test intersection of bounding box with ray.
        const Ray &qray = query.first;
        bool ray_bounds = intersect_ray_aabb (qray, space_bounds);

        return in_bounds_field && ray_bounds;
      }
    };

    // Get solution when close enough: Iterate using Newton's method.
    struct FGetSolution
    {
      DRAY_EXEC bool operator() (stats::Stats &stats,
                                 const QueryT &query,
                                 const ElemPair &elem,
                                 const RefBoxT &ref_box,
                                 SolT &solution)
      {
        Vec<Float, 3> sol_ref_coords = ref_box.template center<Float> ();
        Float sol_ray_dist = query.first.m_near;

        bool sol_found =
        intersect_local (stats, elem.first, elem.second, query.first,
                         query.second, sol_ref_coords, sol_ray_dist, true);
        solution[0] = sol_ref_coords[0]; // Pack
        solution[1] = sol_ref_coords[1];
        solution[2] = sol_ref_coords[2];
        solution[3] = sol_ray_dist;

        return sol_found;
      }
    };

    // Initiate subdivision search.
    SolT solution;
    uint32 ret_code;
    int32 num_solutions =
    SubdivisionSearch::subdivision_search<QueryT, ElemPair, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget> (
    ret_code, stats, ray_iso_query, element, tol_refbox, &domain, &solution, 1);

    ref_coords[0] = solution[0]; // Unpack
    ref_coords[1] = solution[1];
    ref_coords[2] = solution[2];
    ray_dist = solution[3];

    return num_solutions > 0;
  }
  //
  // Returns true if an intersection was found.
  DRAY_EXEC static bool intersect_local (stats::Stats &stats,
                                         const ElemT &mesh_elem,
                                         const FieldOn<ElemT, 1u> &field_elem,
                                         const Ray &ray,
                                         Float isoval,
                                         Vec<Float, 3> &ref_coords,
                                         Float &ray_dist,
                                         bool use_init_guess = false)
  {
    // TODO would be nicer as a lambda.

    // Newton step to solve the in-element isosurface intersection problem.
    struct Stepper
    {
      DRAY_EXEC typename IterativeMethod::StepStatus operator() (Vec<Float, 3 + 1> &xt) const
      {
        Vec<Float, 3> &x = *(Vec<Float, 3> *)&xt[0];
        Float &rdist = *(Float *)&xt[3];

        // Space jacobian and spatial residual.
        Vec<Float, 3> delta_y;
        Vec<Vec<Float, 3>, 3> j_col;
        delta_y = m_transf.eval_d (x, j_col);
        delta_y = m_ray_orig + m_ray_dir * rdist - delta_y;

        // Field gradient and field residual.
        Vec<Float, 1> _delta_f;
        Float &delta_f = _delta_f[0];
        Vec<Vec<Float, 1>, 3> _grad_f;
        Vec<Float, 3> &grad_f = *(Vec<Float, 3> *)&_grad_f[0];
        _delta_f = m_field.eval_d (x, _grad_f);
        delta_f = m_isovalue - delta_f;

        // Inverse of Jacobian (LU decomposition).
        bool inverse_valid;
        Matrix<Float, 3, 3> jacobian;
        for (int32 rdim = 0; rdim < 3; rdim++)
          jacobian.set_col (rdim, j_col[rdim]);
        MatrixInverse<Float, 3> jac_inv (jacobian, inverse_valid);

        // Compute adjustments using first-order approximation.
        Vec<Float, 3> delta_x = jac_inv * delta_y;
        Vec<Float, 3> delta_x_r = jac_inv * m_ray_dir;

        Float delta_r = (delta_f - dot (grad_f, delta_x)) / dot (grad_f, delta_x_r);
        delta_x = delta_x + delta_x_r * delta_r;

        if (!inverse_valid) return IterativeMethod::Abort;

        // Apply the step.
        x = x + delta_x;
        rdist = rdist + delta_r;
        return IterativeMethod::Continue;
      }

      ElemT m_transf;
      FieldOn<ElemT, 1u> m_field;
      Vec<Float, 3> m_ray_orig;
      Vec<Float, 3> m_ray_dir;
      Float m_isovalue;
    } stepper{ mesh_elem, field_elem, ray.m_orig, ray.m_dir, isoval };

    Vec<Float, 4> vref_coords{ ref_coords[0], ref_coords[1], ref_coords[2], ray_dist };
    if (!use_init_guess)
      for (int32 d = 0; d < 3; d++) // Always use the ray_dist coordinate, for now.
        vref_coords[d] = 0.5;

    // TODO somewhere else in the program, figure out how to set the precision
    const Float tol_ref = 1e-4;
    const int32 max_steps = 10;

    // Find solution.
    bool converged = (IterativeMethod::solve (stats, stepper, vref_coords, max_steps,
                                              tol_ref) == IterativeMethod::Converged);

    ref_coords = { vref_coords[0], vref_coords[1], vref_coords[2] };
    ray_dist = vref_coords[3];

    return (converged && mesh_elem.is_inside (ref_coords) &&
            ray.m_near <= ray_dist && ray_dist < ray.m_far);
  }
};

} // namespace dray
#endif

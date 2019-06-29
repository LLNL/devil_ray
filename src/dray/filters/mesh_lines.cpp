#include <dray/filters/mesh_lines.hpp>
#include <dray/array_utils.hpp>
#include <dray/ref_point.hpp>
#include <dray/shaders.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/high_order_intersection.hpp>

#include <dray/high_order_shape.hpp>  // candidate lists.

namespace dray
{

  template Array<RefPoint<float32,3>> intersect_mesh_faces(Array<Ray<float32>> rays, const Mesh<float32> &mesh, const BVH &bvh);
  template Array<RefPoint<float64,3>> intersect_mesh_faces(Array<Ray<float64>> rays, const Mesh<float64> &mesh, const BVH &bvh);

  template Array<Vec<float32,4>> mesh_lines<float32>(Array<Ray<float32>> rays, const Mesh<float32> &mesh, const BVH &bvh);
  template Array<Vec<float32,4>> mesh_lines<float64>(Array<Ray<float64>> rays, const Mesh<float64> &mesh, const BVH &bvh);



  template <typename T>
  Array<RefPoint<T,3>> intersect_mesh_faces(Array<Ray<T>> rays, const Mesh<T> &mesh, const BVH &bvh)
  {
    constexpr int32 ref_dim = 3;

    // Initialize rpoints to same size as rays, each rpoint set to invalid_refpt.
    Array<RefPoint<T,3>> rpoints;
    rpoints.resize(rays.size());
    const RefPoint<T,ref_dim> invalid_refpt{ -1, {-1,-1,-1} };
    array_memset(rpoints, invalid_refpt);

    // Duplicated from MeshField::intersect_mesh_boundary().

    const Vec<T,3> element_guess = {0.5, 0.5, 0.5};
    const T ray_guess = 1.0;

    // Get intersection candidates for all active rays.
    constexpr int32 max_candidates = 64;
    Array<int32> candidates = detail::candidate_ray_intersection<T, max_candidates> (rays, bvh);
    const int32 *candidates_ptr = candidates.get_device_ptr_const();

    const int32 size = rays.size();

      // Define pointers for RAJA kernel.
    MeshAccess<T> device_mesh = mesh.access_device_mesh();
    Ray<T> *ray_ptr = rays.get_device_ptr();
    RefPoint<T,ref_dim> *rpoints_ptr = rpoints.get_device_ptr();

#ifdef DRAY_STATS
    //TODO
    /// stats::AppStatsAccess device_appstats = stats.get_device_appstats();
#endif

    // For each active ray, loop through candidates until found an intersection.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
    {
      Ray<T> &ray = ray_ptr[i];
      RefPoint<T,ref_dim> &rpt = rpoints_ptr[i];

      Vec<T,3> ref_coords = element_guess;
      T ray_dist = ray_guess;

      // In case no intersection is found.
      ray.m_active = 0;
      ray.m_dist = infinity<T>();  // TODO change comparisons for valid rays to check both near and far.

      bool found_inside = false;
      int32 candidate_idx = 0;
      int32 el_idx = candidates_ptr[i*max_candidates + candidate_idx];
      int32 steps_taken = 0;
      while (!found_inside && candidate_idx < max_candidates && el_idx != -1)
      {
        ref_coords = element_guess;
        ray_dist = ray_guess;
        const bool use_init_guess = true;

#ifdef DRAY_STATS
        stats::IterativeProfile iter_prof;    iter_prof.construct();

        MeshElem<T> mesh_elem = device_mesh.get_elem(el_idx);
        found_inside = Intersector_RayFace<T>::intersect(iter_prof, mesh_elem, ray,
            ref_coords, ray_dist, use_init_guess);

        //TODO
        /// steps_taken = iter_prof.m_num_iter;
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_tests, 1);
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_test_iterations, steps_taken);
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_tests, 1);
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_test_iterations, steps_taken);
#else

        //TODO after add this to the Intersector_RayFace interface.
        /// found_inside = Intersector_RayFace<T>::intersect(device_mesh.get_elem(el_idx), ray,
        ///     ref_coords, ray_dist, use_init_guess);
#endif
        if (found_inside && ray_dist < ray.m_dist && ray_dist >= ray.m_near)
        {
          rpt.m_el_id = el_idx;
          rpt.m_el_coords = ref_coords;
          ray.m_dist = ray_dist;
        }

        // Continue searching with the next candidate.
        candidate_idx++;
        el_idx = candidates_ptr[i*max_candidates + candidate_idx];

      } // end while

  #ifdef DRAY_STATS
      if (found_inside)
      {
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_hits, 1);
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_hit_iterations, steps_taken);
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_hits, 1);
        /// RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_hit_iterations, steps_taken);
      }
  #endif
    });  // end RAJA

    return rpoints;
  }

  template <typename T>
  Array<Vec<float32,4>> mesh_lines(Array<Ray<T>> rays, const Mesh<T,3> &mesh, const BVH &bvh)
  {
    using Color = Vec<float32,4>;
    constexpr int32 ref_dim = 3;

    Array<Color> color_buffer;
    color_buffer.resize(rays.size());

    // Initialize the color buffer to (0,0,0,0).
    const Color init_color = make_vec4f(0.f, 0.f, 0.f, 0.f);
    const Color bg_color   = make_vec4f(1.f, 1.f, 1.f, 1.f);
    array_memset_vec(color_buffer, init_color);

    // Initialize fragment shader.
    ShadeMeshLines shader;
    const Color face_color = make_vec4f(0.f, 0.f, 0.f, 1.f);
    const Color line_color = make_vec4f(1.f, 1.f, 1.f, 1.f);
    const float32 line_ratio = 0.05;
    shader.set_uniforms(line_color, face_color, line_ratio);

    // Start the rays out at the min distance from calc ray start.
    // Note: Rays that have missed the mesh bounds will have near >= far,
    //       so after the copy, we can detect misses as dist >= far.
    dray::AABB<3> mesh_bounds = mesh.get_bounds();
    calc_ray_start(rays, mesh_bounds);
    Array<int32> active_rays = active_indices(rays);

    // Remove the rays which totally miss the mesh.
    rays = gather(rays, active_rays);

    //TODO where does this go?
/// #ifdef DRAY_STATS
///   std::shared_ptr<stats::AppStats> app_stats_ptr = stats::global_app_stats.get_shared_ptr();
///   app_stats_ptr->m_query_stats.resize(rays.size());
///   app_stats_ptr->m_elem_stats.resize(m_size_el);
///
///   stats::AppStatsAccess device_appstats = app_stats_ptr->get_device_appstats();
///   RAJA::forall<for_policy>(RAJA::RangeSegment(0, rays.size()), [=] DRAY_LAMBDA (int32 ridx)
///   {
///     device_appstats.m_query_stats_ptr[ridx].construct();
///   });
///   RAJA::forall<for_policy>(RAJA::RangeSegment(0, m_size_el), [=] DRAY_LAMBDA (int32 el_idx)
///   {
///     device_appstats.m_elem_stats_ptr[el_idx].construct();
///   });
/// #endif

    Array<RefPoint<T,ref_dim>> rpoints = intersect_mesh_faces(rays, mesh, bvh);
    Color *color_buffer_ptr = color_buffer.get_device_ptr();
    const RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr_const();
    const Ray<T> *rays_ptr = rays.get_device_ptr_const();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, rpoints.size()), [=] DRAY_LAMBDA (int32 ii)
    {
      const RefPoint<T> &rpt = rpoints_ptr[ii];
      if (rpt.m_el_id != -1)
      {
        Color pixel_color = shader(rpt.m_el_coords);
        color_buffer_ptr[rays_ptr[ii].m_pixel_id] = pixel_color;
      }
    });

    Shader::composite_bg(color_buffer, bg_color);

    return color_buffer;

  }

template<typename T>
Array<Vec<float32,4>>
Pseudocolor::execute(Array<Ray<T>> &rays, DataSet<T> &data_set)
{
  //return mesh_lines(Array<Ray<T>> rays, data_set.get_mesh(), const BVH &bvh);
}

};//naemespace dray

#include <dray/Vis/mesh_lines.hpp>
#include <dray/array_utils.hpp>
#include <dray/ref_point.hpp>
#include <dray/shaders.hpp>
#include <dray/linear_bvh_builder.hpp>

namespace dray
{
  
  template Array<Vec<float32,4>> mesh_lines<float32>(Array<Ray<float32>> rays, const Mesh<float32> &mesh);
  template Array<Vec<float32,4>> mesh_lines<float64>(Array<Ray<float64>> rays, const Mesh<float64> &mesh);

  template <typename T>
  Array<Vec<float32,4>> mesh_lines(Array<Ray<T>> rays, const Mesh<T,3> &mesh)
  {
    using Color = Vec<float32,4>;
    constexpr int32 ref_dim = 3;

    Array<Color> color_buffer;
    color_buffer.resize(rays.size());

    // Initialize the color buffer to (0,0,0,0).
    const Color init_color = make_vec4f(0.f, 0.f, 0.f, 0.f);
    const Color bg_color   = make_vec4f(1.f, 1.f, 1.f, 1.f);
    const Color face_color = make_vec4f(0.f, 0.f, 0.f, 0.f);
    const Color line_color = make_vec4f(1.f, 1.f, 1.f, 1.f);
    array_memset_vec(color_buffer, init_color);

    // Start the rays out at the min distance from calc ray start.
    // Note: Rays that have missed the mesh bounds will have near >= far,
    //       so after the copy, we can detect misses as dist >= far.
    dray::AABB<3> mesh_bounds = dray::reduce(mesh.get_aabbs());  // more direct way.
    calc_ray_start(rays, mesh_bounds);
    Array<int32> active_rays = active_indices(rays);// Remove the rays which totally miss the mesh.

    //TODO
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

    Array<RefPoint<T,ref_dim>> rpoints;
    rpoints.resize(rays.size());

    const RefPoint<T,ref_dim> invalid_refpt{ -1, {-1,-1,-1} };

    int32 dbg_count_iter = 0;
    std::cout<<"active rays "<<active_rays.size()<<"\n";
    while(active_rays.size() > 0)
    {
      Array<Vec<T,3>> wpoints = calc_tips(rays);

      array_memset(rpoints, invalid_refpt);

      // Find elements and reference coordinates for the points.
      //TODO use face intersector.  Maybe look at isosurface shading method as example.


      //TODO something like this
/// #ifdef DRAY_STATS
///       locate(active_rays, wpoints, rpoints, *app_stats_ptr);
/// #else
///       locate(active_rays, wpoints, rpoints);
/// #endif

      //TODO
      /// // Retrieve shading information at those points (scalar field value, gradient).
      /// Array<ShadingContext<T>> shading_ctx = get_shading_context(rays, rpoints);

      /// // shade and blend sample using shading context  with color buffer
      /// Shader::blend(color_buffer, shading_ctx);

      /// advance_ray(rays, sample_dist);

      /// active_rays = active_indices(rays);

      /// std::cout << "MeshField::integrate() - Finished iteration " << dbg_count_iter++ << std::endl;
    }

    Shader::composite_bg(color_buffer,bg_color);

    return color_buffer;

  }

};//naemespace dray

#include <dray/Vis/mesh_lines.hpp>
#include <dray/array_utils.hpp>
#include <dray/ref_point.hpp>
#include <dray/shaders.hpp>
#include <dray/linear_bvh_builder.hpp>

namespace dray
{
  
  template Array<RefPoint<float32,3>> intersect_mesh_faces(Array<Ray<float32>> rays, const Mesh<float32> &mesh);
  template Array<RefPoint<float64,3>> intersect_mesh_faces(Array<Ray<float64>> rays, const Mesh<float64> &mesh);

  template Array<Vec<float32,4>> mesh_lines<float32>(Array<Ray<float32>> rays, const Mesh<float32> &mesh);
  template Array<Vec<float32,4>> mesh_lines<float64>(Array<Ray<float64>> rays, const Mesh<float64> &mesh);



  template <typename T>
  Array<RefPoint<T,3>> intersect_mesh_faces(Array<Ray<T>> rays, const Mesh<T> &mesh)
  {
    constexpr int32 ref_dim = 3;

    // Initialize rpoints to same size as rays, each rpoint set to invalid_refpt.
    Array<RefPoint<T,3>> rpoints;
    rpoints.resize(rays.size());
    const RefPoint<T,ref_dim> invalid_refpt{ -1, {-1,-1,-1} };
    array_memset(rpoints, invalid_refpt);

    // TODO follow isosurface example to find intersection.

    return rpoints;
  }

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
    array_memset_vec(color_buffer, init_color);

    // Initialize fragment shader.
    ShadeMeshLines shader;
    const Color face_color = make_vec4f(0.f, 0.f, 0.f, 0.f);
    const Color line_color = make_vec4f(1.f, 1.f, 1.f, 1.f);
    const float32 line_ratio = 0.05;
    shader.set_uniforms(line_color, face_color, line_ratio);

    // Start the rays out at the min distance from calc ray start.
    // Note: Rays that have missed the mesh bounds will have near >= far,
    //       so after the copy, we can detect misses as dist >= far.
    dray::AABB<3> mesh_bounds = dray::reduce(mesh.get_aabbs());  // more direct way.
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

    Array<RefPoint<T,ref_dim>> rpoints = intersect_mesh_faces(rays, mesh);
    Color *color_buffer_ptr = color_buffer.get_device_ptr();
    const RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr_const();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, rpoints.size()), [=] DRAY_LAMBDA (int32 ii)
    {
      color_buffer_ptr[ii] = shader(rpoints_ptr[ii].m_el_coords);
    });

    Shader::composite_bg(color_buffer, bg_color);

    return color_buffer;

  }

};//naemespace dray

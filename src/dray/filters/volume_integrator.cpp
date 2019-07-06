#include <dray/filters/volume_integrator.hpp>
#include <dray/filters/internal/get_shading_context.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>

#include <assert.h>

namespace dray
{

VolumeIntegrator::VolumeIntegrator()
  : m_color_table("ColdAndHot"),
    m_num_samples(100)
{
  m_color_table.add_alpha(0.0000, .1f);
  m_color_table.add_alpha(1.0000, .2f);
}

template<typename T>
Array<Vec<float32,4>>
VolumeIntegrator::execute(Array<Ray<T>> &rays,
                          DataSet<T> &data_set)
{
  Mesh<T,3> mesh = data_set.get_mesh();

  assert(m_field_name != "");
  dray::Shader::set_color_table(m_color_table);

  Field<T> field = data_set.get_field(m_field_name);

  dray::AABB<> bounds = mesh.get_bounds();
  dray::float32 mag = (bounds.max() - bounds.min()).magnitude();
  const float32 sample_dist = mag / dray::float32(m_num_samples);


  calc_ray_start(rays, mesh.get_bounds());

  const int32 num_elems = mesh.get_num_elem();

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());

  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  Vec<float32,4> bg_color = make_vec4f(1.f,1.f,1.f,1.f);

  array_memset_vec(color_buffer, init_color);

  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  Array<int32> active_rays = active_indices(rays);

#ifdef DRAY_STATS
  std::shared_ptr<stats::AppStats> app_stats_ptr = stats::global_app_stats.get_shared_ptr();

  app_stats_ptr->m_query_stats.resize(rays.size());
  app_stats_ptr->m_elem_stats.resize(num_elems);

  stats::AppStatsAccess device_appstats = app_stats_ptr->get_device_appstats();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, rays.size()), [=] DRAY_LAMBDA (int32 ridx)
  {
    device_appstats.m_query_stats_ptr[ridx].construct();
  });

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 el_idx)
  {
    device_appstats.m_elem_stats_ptr[el_idx].construct();
  });
#endif

  Array<RefPoint<T,3>> rpoints;
  rpoints.resize(rays.size());

  const RefPoint<T,3> invalid_refpt{ -1, {-1,-1,-1} };

  while(active_rays.size() > 0)
  {
    Array<Vec<T,3>> wpoints = calc_tips(rays);

    array_memset(rpoints, invalid_refpt);

    // Find elements and reference coordinates for the points.
#ifdef DRAY_STATS
    mesh.locate(active_rays, wpoints, rpoints, *app_stats_ptr);
#else
    mesh.locate(active_rays, wpoints, rpoints);
#endif
    // Retrieve shading information at those points (scalar field value, gradient).
    Array<ShadingContext<T>> shading_ctx =
      internal::get_shading_context(rays, field, mesh, rpoints);

    // shade and blend sample using shading context  with color buffer
    Shader::blend(color_buffer, shading_ctx);

    advance_ray(rays, sample_dist);

    active_rays = active_indices(rays);

  }

  Shader::composite_bg(color_buffer,bg_color);

  return color_buffer;
}

void
VolumeIntegrator::set_field(const std::string field_name)
{
 m_field_name = field_name;
}

void
VolumeIntegrator::set_color_table(const ColorTable &color_table)
{
  m_color_table = color_table;
}

void
VolumeIntegrator::set_num_samples(const int32 num_samples)
{
  assert(num_samples > 0);
  m_num_samples = num_samples;
}

template
Array<Vec<float32,4>>
VolumeIntegrator::execute<float32>(Array<Ray<float32>> &rays,
                                   DataSet<float32> &data_set);

template
Array<Vec<float32,4>>
VolumeIntegrator::execute<float64>(Array<Ray<float64>> &rays,
                                   DataSet<float64> &data_set);

}//namespace dray


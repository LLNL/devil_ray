#include <dray/filters/volume_integrator.hpp>
#include <dray/filters/internal/get_fragments.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

} // namespace detail

VolumeIntegrator::VolumeIntegrator()
  : m_color_table("ColdAndHot"),
    m_num_samples(100)
{
  m_color_table.add_alpha(0.0000, .1f);
  m_color_table.add_alpha(1.0000, .2f);
}

template<class ElemT>
void
VolumeIntegrator::execute(Array<Ray> &rays,
                          DataSet<ElemT> &data_set,
                          Framebuffer &fb)
{
  Mesh<ElemT> mesh = data_set.get_mesh();

  assert(m_field_name != "");

  constexpr float32 correction_scalar = 10.f;
  float32 ratio = correction_scalar / m_num_samples;
  dray::Shader::set_color_table(m_color_table.correct_opacity(ratio));

  Field<FieldOn<ElemT, 1u>> field = data_set.get_field(m_field_name);

  dray::AABB<> bounds = mesh.get_bounds();
  dray::float32 mag = (bounds.max() - bounds.min()).magnitude();
  const float32 sample_dist = mag / dray::float32(m_num_samples);


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
  cull_missed_rays(rays, mesh.get_bounds());


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

  Array<RefPoint<3>> rpoints;
  rpoints.resize(rays.size());

  const RefPoint<3> invalid_refpt{ -1, {-1,-1,-1} };
  // Hack to try to advance inside volume
  // TODO: cacl actual ray start and end
  advance_ray(rays, sample_dist);

  const int32 ray_size = rays.size();
  const Ray *rays_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 i)
  {
    const Ray ray = rays_ptr[i];

  });

#if 0
  while(active_rays.size() > 0)
  {
    Array<Vec<Float,3>> wpoints = calc_tips(rays);

    array_memset(rpoints, invalid_refpt);

    // Find elements and reference coordinates for the points.
#ifdef DRAY_STATS
    mesh.locate(active_rays, wpoints, rpoints, *app_stats_ptr);
#else
    mesh.locate(active_rays, wpoints, rpoints);
#endif
    // Retrieve shading information at those points (scalar field value, gradient).
    Array<ShadingContext> shading_ctx =
      internal::get_shading_context(rays, field.get_range(), field, mesh, rpoints);

    // shade and blend sample using shading context  with color buffer
    Shader::blend(color_buffer, shading_ctx);

    advance_ray(rays, sample_dist);

    active_rays = active_indices(rays);

  }
#endif
  Shader::composite_bg(color_buffer,bg_color);
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

using Hex = MeshElem<3u, ElemType::Quad, Order::General>;
template
void
VolumeIntegrator::execute<Hex>(Array<Ray> &rays,
                               DataSet<Hex> &data_set,
                               Framebuffer &fb);

}//namespace dray


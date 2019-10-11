#include <dray/mfem_volume_integrator.hpp>

#include <dray/mfem_grid_function.hpp>
#include <dray/shading_context.hpp>
#include <dray/color_table.hpp>

#include <dray/array_utils.hpp>
#include <dray/policies.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

namespace dray
{

namespace detail
{

// Binary functor IsLess
//
template <typename T>
struct IsLess
{
  DRAY_EXEC bool operator()(const T &dist, const T &far) const
  {
    return dist < far;
  }
};

} // namespace detail

MFEMVolumeIntegrator::MFEMVolumeIntegrator()
  : m_mesh(NULL),
    m_field(NULL)
{
  //if this ever happens this will segfault
  //this is private so that should not happen
}

MFEMVolumeIntegrator::MFEMVolumeIntegrator(MFEMMesh &mesh, MFEMGridFunction &gf)
  : m_mesh(mesh),
    m_field(gf)
{
  AABB<> bounds = m_mesh.get_bounds();

  float32 lx = bounds.m_ranges[0].length();
  float32 ly = bounds.m_ranges[1].length();
  float32 lz = bounds.m_ranges[2].length();

  float32 mag = sqrt(lx*lx + ly*ly + lz*lz);

  constexpr int num_samples = 300;


  m_sample_dist = mag / float32(num_samples);

}

MFEMVolumeIntegrator::~MFEMVolumeIntegrator()
{

}

Array<Vec<float32,4>>
MFEMVolumeIntegrator::integrate(Array<Ray> rays)
{
  DRAY_LOG_OPEN("mfem_volume_integrate");

  Timer tot_time;

  calc_ray_start(rays, m_mesh.get_bounds());

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
  //Array<int32> active_rays = array_counting(rays.size(),0,1);
  Array<int32> active_rays = active_indices(rays);
  while(active_rays.size() > 0)
  {
    Timer timer;
    Timer step_timer;

    std::cout<<"active rays "<<active_rays.size()<<"\n";
    // Find elements and reference coordinates for the points.
    Array<Vec<Float,3>> points = calc_tips(rays);

    m_mesh.locate(points, active_rays, rays);
    DRAY_LOG_ENTRY("locate", timer.elapsed());
    timer.reset();

    // Retrieve shading information at those points (scalar field value, gradient).
    Array<ShadingContext> shading_ctx = m_field.get_shading_context(rays);
    DRAY_LOG_ENTRY("get_shading_context", timer.elapsed());
    timer.reset();

    // shade and blend sample using shading context  with color buffer
    Shader::blend(color_buffer, shading_ctx);
    DRAY_LOG_ENTRY("blend", timer.elapsed());
    timer.reset();

    // advance the rays
    advance_ray(rays, m_sample_dist);
    DRAY_LOG_ENTRY("advance_ray", timer.elapsed());
    timer.reset();

    // compact remaining active rays
    active_rays = active_indices(rays);

    DRAY_LOG_ENTRY("compact_rays", timer.elapsed());
    timer.reset();

    DRAY_LOG_ENTRY("step_time", step_timer.elapsed());
  }

  Shader::composite_bg(color_buffer,bg_color);
  DRAY_LOG_ENTRY("integrate_time", tot_time.elapsed());
  DRAY_LOG_CLOSE();

  return color_buffer;
}

} // namespace dray

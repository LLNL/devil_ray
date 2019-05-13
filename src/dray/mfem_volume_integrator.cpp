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
//
// utility function to find estimated intersection with the bounding
// box of the mesh
//
// After calling:
//   rays m_near   : set to estimated mesh entry
//   rays m_far    : set to estimated mesh exit
//   rays hit_idx  : -1 if ray missed the AABB and 1 if it hit
//
//   if ray missed then m_far <= m_near, if ray hit then m_far > m_near.
//
template<typename T>
void calc_ray_start(Ray<T> &rays, AABB bounds)
{
  // avoid lambda capture issues
  AABB mesh_bounds = bounds;

  const Vec<T,3> *dir_ptr = rays.m_dir.get_device_ptr_const();
  const Vec<T,3> *orig_ptr = rays.m_orig.get_device_ptr_const();

  T *near_ptr = rays.m_near.get_device_ptr();
  T *far_ptr  = rays.m_far.get_device_ptr();
  int32 *hit_idx_ptr = rays.m_hit_idx.get_device_ptr();

  const int32 size = rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<T,3> ray_dir = dir_ptr[i];
    Vec<T,3> ray_orig = orig_ptr[i];

    float32 dirx = static_cast<float32>(ray_dir[0]);
    float32 diry = static_cast<float32>(ray_dir[1]);
    float32 dirz = static_cast<float32>(ray_dir[2]);
    float32 origx = static_cast<float32>(ray_orig[0]);
    float32 origy = static_cast<float32>(ray_orig[1]);
    float32 origz = static_cast<float32>(ray_orig[2]);

    float32 inv_dirx = rcp_safe(dirx);
    float32 inv_diry = rcp_safe(diry);
    float32 inv_dirz = rcp_safe(dirz);

    float32 odirx = origx * inv_dirx;
    float32 odiry = origy * inv_diry;
    float32 odirz = origz * inv_dirz;

    float32 xmin = mesh_bounds.m_x.min() * inv_dirx - odirx;
    float32 ymin = mesh_bounds.m_y.min() * inv_diry - odiry;
    float32 zmin = mesh_bounds.m_z.min() * inv_dirz - odirz;
    float32 xmax = mesh_bounds.m_x.max() * inv_dirx - odirx;
    float32 ymax = mesh_bounds.m_y.max() * inv_diry - odiry;
    float32 zmax = mesh_bounds.m_z.max() * inv_dirz - odirz;

    const float32 min_int = 0.f;
    float32 min_dist = max(max(max(min(ymin, ymax), min(xmin, xmax)), min(zmin, zmax)), min_int);
    float32 max_dist = min(min(max(ymin, ymax), max(xmin, xmax)), max(zmin, zmax));

    int32 hit = -1; // miss flag

    if (max_dist > min_dist)
    {
      hit = 1; 
    }

    near_ptr[i] = min_dist;
    far_ptr[i] = max_dist;
    
    hit_idx_ptr[i] = hit;

  });
}

// this is a place holder function. Normally we could just set 
// values somewhere else, but for testing lets just do this now
template<typename T>
void advance_ray(Ray<T> &rays, float32 distance)
{
  // aviod lambda capture issues
  T dist = distance;

  /// const T *near_ptr = rays.m_near.get_device_ptr_const();
  /// const T *far_ptr  = rays.m_far.get_device_ptr_const();
  const int32 *active_ray_ptr = rays.m_active_rays.get_device_ptr_const();

  T *dist_ptr  = rays.m_dist.get_device_ptr();
  /// int32 *hit_idx_ptr = rays.m_hit_idx.get_device_ptr();

  const int32 size = rays.m_active_rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 ray_idx = active_ray_ptr[i];
    /// T far = far_ptr[ray_idx];
    T current_dist = dist_ptr[ray_idx];
    // advance ray
    current_dist += dist;
    dist_ptr[ray_idx] = current_dist;
     
  });
}


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
  AABB bounds = m_mesh.get_bounds();

  float32 lx = bounds.m_x.length();
  float32 ly = bounds.m_y.length();
  float32 lz = bounds.m_z.length();
  
  float32 mag = sqrt(lx*lx + ly*ly + lz*lz);

  constexpr int num_samples = 300;
  

  m_sample_dist = mag / float32(num_samples);

}

MFEMVolumeIntegrator::~MFEMVolumeIntegrator()
{

}
  
template<typename T>
Array<Vec<float32,4>>
MFEMVolumeIntegrator::integrate(Ray<T> rays)
{
  DRAY_LOG_OPEN("mfem_volume_integrate");

  Timer tot_time; 
  
  detail::calc_ray_start(rays, m_mesh.get_bounds());

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());
  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  Vec<float32,4> bg_color = make_vec4f(1.f,1.f,1.f,1.f);
  array_memset_vec(color_buffer, init_color);

  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.
  array_copy(rays.m_dist, rays.m_near);

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  //Array<int32> active_rays = array_counting(rays.size(),0,1);
  rays.m_active_rays = compact(rays.m_active_rays, rays.m_dist, rays.m_far, detail::IsLess<T>());

  while(rays.m_active_rays.size() > 0) 
  {
    Timer timer; 
    Timer step_timer; 

    std::cout<<"active rays "<<rays.m_active_rays.size()<<"\n"; 
    // Find elements and reference coordinates for the points.
    m_mesh.locate(rays.calc_tips(), rays.m_active_rays, rays.m_hit_idx, rays.m_hit_ref_pt);
    DRAY_LOG_ENTRY("locate", timer.elapsed());
    timer.reset();

    // Retrieve shading information at those points (scalar field value, gradient).
    ShadingContext<T> shading_ctx = m_field.get_shading_context(rays);
    DRAY_LOG_ENTRY("get_shading_context", timer.elapsed());
    timer.reset();

    // shade and blend sample using shading context  with color buffer
    Shader::blend(color_buffer, shading_ctx);
    DRAY_LOG_ENTRY("blend", timer.elapsed());
    timer.reset();

    // advance the rays
    detail::advance_ray(rays, m_sample_dist); 
    DRAY_LOG_ENTRY("advance_ray", timer.elapsed());
    timer.reset();

    // compact remaining active rays 
    rays.m_active_rays = compact(rays.m_active_rays, rays.m_dist, rays.m_far, detail::IsLess<T>());
    DRAY_LOG_ENTRY("compact_rays", timer.elapsed());
    timer.reset();

    DRAY_LOG_ENTRY("step_time", step_timer.elapsed());
  }
  
  Shader::composite_bg(color_buffer,bg_color);
  DRAY_LOG_ENTRY("integrate_time", tot_time.elapsed());
  DRAY_LOG_CLOSE();

  return color_buffer;
}
  
// explicit instantiations
template Array<Vec<float32,4>> MFEMVolumeIntegrator::integrate(ray32 rays);
template Array<Vec<float32,4>> MFEMVolumeIntegrator::integrate(ray64 rays);

} // namespace dray

#include <dray/mfem_volume_integrator.hpp>

#include <dray/mfem_grid_function.hpp>

#include <dray/array_utils.hpp>
#include <dray/policies.hpp>

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
template<typename T>
void calc_ray_start(Ray<T> &rays, AABB bounds)
{
  /// AABB mesh_bounds;
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
      near_ptr[i] = min_dist;
      far_ptr[i] = max_dist;
    }

    hit_idx_ptr[i] = hit;

  });
}

} // namespace detail

MFEMVolumeIntegrator::MFEMVolumeIntegrator()
  : m_mesh(NULL)
{
  //if this ever happens this will segfault
  //this is private so that should not happen
}

MFEMVolumeIntegrator::MFEMVolumeIntegrator(MFEMMesh &mesh)
  : m_mesh(mesh)
{
  AABB bounds = m_mesh.get_bounds();

  float32 lx = bounds.m_x.length();
  float32 ly = bounds.m_y.length();
  float32 lz = bounds.m_z.length();
  
  float32 mag = sqrt(lx*lx + ly*ly + lz*lz);

  constexpr int num_samples = 200;

  m_sample_dist = mag / float32(num_samples);

}

MFEMVolumeIntegrator::~MFEMVolumeIntegrator()
{

}
  
template<typename T>
void 
MFEMVolumeIntegrator::integrate(Ray<T> &rays, const MFEMGridFunction &scalarField)
{
  detail::calc_ray_start(rays, m_mesh.get_bounds());

  // Get the range of the field.
  float32 field_min, field_max;
  scalarField.get_bounds(field_min, field_max);
  float32 field_range = field_max - field_min;

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());
  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  array_memset_vec(color_buffer, init_color);

  // Start the rays out at the min distance from calc ray start.
  array_copy(rays.m_dist, rays.m_near);

  // Initialize the set of 'active rays'.
  Array<int32> active_ray_idxs;
  int32 num_active_rays;
  //TODO COMPACTOR(active_ray_idxs, num_active_rays,  rays.m_dist, rays.m_far, [] (T dist, T far) { return dist < far });
  //Assuming that COMPACTOR will resize active_ray_idxs, and set num_active_rays.
  //Parameters 3 and 4 are [in] params.
  //Parameter 5 is a lambda.
  //Not sure how your COMPACTOR will be structured...

  // Local variables to pacify RAJA + lambdas.
  const float32 sample_dist = m_sample_dist;

  //const Vec<T,3> *dir_ptr = rays.m_dir.get_device_ptr_const();
  //const Vec<T,3> *orig_ptr = rays.m_orig.get_device_ptr_const();

  while (num_active_rays > 0)
  {

    //T *dist_ptr = rays.m_dist.get_device_ptr();

    //    locate_points( ray.dist * dir + orig)
    //    get shading context (scalar + normal(gradient))
    //    shade and blend sample using shading context  with color buffer
    //    advance ray.distance += sammple_distance

    // Update the set of 'active rays'.
    //TODO COMPACTOR(active_ray_idxs, num_active_rays,  rays.m_dist, rays.m_far, [] (T dist, T far) { return dist < far });
  }
}
  
// explicit instantiations
template void MFEMVolumeIntegrator::integrate(ray32 &rays, const MFEMGridFunction &scalarField);
template void MFEMVolumeIntegrator::integrate(ray64 &rays, const MFEMGridFunction &scalarField);

} // namespace dray

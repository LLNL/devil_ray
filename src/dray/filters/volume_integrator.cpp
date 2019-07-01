#include <dray/filters/volume_integrator.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

template <typename T>
Array<ShadingContext<T>>
get_shading_context(Array<Ray<T>> &rays,
                    Field<T> &field,
                    Mesh<T> &mesh,
                    Array<RefPoint<T,3>> &rpoints)
{
  // Ray (read)    RefPoint (read)      ShadingContext (write)
  // ---           -----             --------------
  // m_pixel_id    m_el_id           m_pixel_id
  // m_dir         m_el_coords       m_ray_dir
  // m_orig
  // m_dist                          m_hit_pt
  //                                 m_is_valid
  //                                 m_sample_val
  //                                 m_normal
  //                                 m_gradient_mag

  //TODO store gradient and jacobian into separate fields??
  //Want to be able to use same get_shading_context() whether
  //the task is to shade isosurface or boundary surface...
  //
  //No, it's probably better to make a different get_shading_context
  //that knows it is using the cross product of two derivatives as the surface normal.
  //One reason is that rays hit_idx should store an index into which
  //the face id is embedded, i.e. (face_id + 6*el_id).
  //
  //In that case, "normal" really means the normal we will use for shading.
  //So it is appropriate to flip the normal to align with view, in this function.
  //(If it didn't mean "normal", but rather "gradient," then we shouldn't flip.)
  const int32 size_rays = rays.size();
  //const int32 size_active_rays = rays.m_active_rays.size();

  Array<ShadingContext<T>> shading_ctx;
  shading_ctx.resize(size_rays);
  ShadingContext<T> *ctx_ptr = shading_ctx.get_device_ptr();

  // Adopt the fields (m_pixel_id) and (m_dir) from rays to intersection_ctx.
  //shading_ctx.m_pixel_id = rays.m_pixel_id;
  //shading_ctx.m_ray_dir = rays.m_dir;

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<T,3> one_two_three = {123., 123., 123.};

  const int32 size = rays.size();

  const Range<> field_range = field.get_range();
  const T field_min = field_range.min();
  const T field_range_rcp = rcp_safe( field_range.length() );

  const Ray<T> *ray_ptr = rays.get_device_ptr_const();
  const RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr_const();

  MeshAccess<T> device_mesh = mesh.access_device_mesh();
  FieldAccess<T> device_field = field.access_device_field();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    ShadingContext<T> ctx;
    // TODO: create struct initializers
    ctx.m_hit_pt = one_two_three;
    ctx.m_normal = one_two_three;
    ctx.m_sample_val = 3.14f;
    ctx.m_gradient_mag = 55.55f;

    const Ray<T> &ray = ray_ptr[i];
    const RefPoint<T,3> &rpt = rpoints_ptr[i];

    ctx.m_pixel_id = ray.m_pixel_id;
    ctx.m_ray_dir  = ray.m_dir;

    if (rpt.m_el_id == -1)
    {
      // There is no intersection.
      ctx.m_is_valid = 0;
    }
    else
    {
      // There is an intersection.
      ctx.m_is_valid = 1;
    }

    if(ctx.m_is_valid)
    {
      // Compute hit point using ray origin, direction, and distance.
      ctx.m_hit_pt = ray.m_orig + ray.m_dir * ray.m_dist;

      // Evaluate element transformation to get scalar field value and gradient.

      const int32 el_id = rpt.m_el_id;
      const Vec<T,3> ref_pt = rpt.m_el_coords;

      Vec<T,3> space_val;
      Vec<Vec<T,3>,3> space_deriv;
      device_mesh.get_elem(el_id).eval(ref_pt, space_val, space_deriv);

      Vec<T,1> field_val;
      Vec<Vec<T,1>,3> field_deriv;
      device_field.get_elem(el_id).eval(ref_pt, field_val, field_deriv);

      // Move derivatives into matrix form.
      Matrix<T,3,3> jacobian;
      Matrix<T,1,3> gradient_h;
      for (int32 rdim = 0; rdim < 3; rdim++)
      {
        jacobian.set_col(rdim, space_deriv[rdim]);
        gradient_h.set_col(rdim, field_deriv[rdim]);
      }

      // Compute spatial field gradient as g = gh * J_inv.
      bool inv_valid;
      const Matrix<T,3,3> j_inv = matrix_inverse(jacobian, inv_valid);
      //TODO How to handle the case that inv_valid == false?
      const Matrix<T,1,3> gradient_mat = gradient_h * j_inv;
      Vec<T,3> gradient = gradient_mat.get_row(0);

      // Output.
      // TODO: deffer this calculation into the shader
      // output actual scalar value and normal
      ctx.m_sample_val = (field_val[0] - field_min) * field_range_rcp;
      ctx.m_gradient_mag = gradient.magnitude();
      gradient.normalize();   //TODO What if the gradient is (0,0,0)?

      if (dot(gradient, ray.m_dir) > 0.0f)
      {
        gradient = -gradient;   //Flip back toward camera.
      }
      ctx.m_normal = gradient;
    }

    ctx_ptr[i] = ctx;

  });

  return shading_ctx;
}
}

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

  constexpr int num_samples = 100;
  dray::AABB<> bounds = mesh.get_bounds();
  dray::float32 mag = (bounds.max() - bounds.min()).magnitude();
  const float32 sample_dist = mag / dray::float32(num_samples);


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

  int32 dbg_count_iter = 0;
  std::cout<<"active rays "<<active_rays.size()<<"\n";
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
      detail::get_shading_context(rays, field, mesh, rpoints);

    // shade and blend sample using shading context  with color buffer
    Shader::blend(color_buffer, shading_ctx);

    advance_ray(rays, sample_dist);

    active_rays = active_indices(rays);

    std::cout << "MeshField::integrate() - Finished iteration "
              << dbg_count_iter++ << std::endl;
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

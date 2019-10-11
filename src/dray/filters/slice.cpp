#include <dray/filters/slice.hpp>
#include <dray/filters/internal/get_shading_context.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

template <class ElemT>
Array<ShadingContext>
get_shading_context_slice(Array<Ray> &rays,
                          Field<FieldOn<ElemT, 1u>> &field,
                          Array<RefPoint<3>> &rpoints,
                          Vec<float32,3> &normal)
{
  const int32 size_rays = rays.size();

  Array<ShadingContext> shading_ctx;
  shading_ctx.resize(size_rays);
  ShadingContext *ctx_ptr = shading_ctx.get_device_ptr();

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<Float,3> one_two_three = {123., 123., 123.};

  const int32 size = rays.size();

  const Range<> field_range = field.get_range();
  const Float field_min = field_range.min();
  const Float field_range_rcp = rcp_safe( field_range.length() );

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const RefPoint<3> *rpoints_ptr = rpoints.get_device_ptr_const();

  FieldAccess<FieldOn<ElemT, 1u>> device_field = field.access_device_field();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    ShadingContext ctx;
    // TODO: create struct initializers
    ctx.m_hit_pt = one_two_three;
    ctx.m_normal = normal;
    ctx.m_sample_val = 3.14f;

    const Ray &ray = ray_ptr[i];
    const RefPoint<3> &rpt = rpoints_ptr[i];

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
      const Vec<Float,3> ref_pt = rpt.m_el_coords;

      Vec<Float,1> field_val;
      Vec<Vec<Float,1>,3> field_deriv;
      field_val = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv);

      // Output.
      // TODO: deffer this calculation into the shader
      // output actual scalar value and normal
      ctx.m_sample_val = (field_val[0] - field_min) * field_range_rcp;
      //printf("m_sample %f\n",ctx.m_sample_val);

      if (dot(ctx.m_normal, ray.m_dir) > 0.0f)
      {
        ctx.m_normal = -ctx.m_normal;   //Flip back toward camera.
      }
    }

    ctx_ptr[i] = ctx;

  });

  return shading_ctx;
}

Array<Vec<Float,3>>
calc_sample_points(Array<Ray> &rays,
                   const Vec<float32,3> &point,
                   const Vec<float32,3> &normal)
{
  const int32 size = rays.size();

  Array<Vec<Float,3>> points;
  points.resize(size);

  Vec<Float,3> t_normal;
  t_normal[0] = normal[0];
  t_normal[1] = normal[1];
  t_normal[2] = normal[2];

  Vec<Float,3> t_point;
  t_point[0] = point[0];
  t_point[1] = point[1];
  t_point[2] = point[2];

  Vec<Float,3> *points_ptr = points.get_device_ptr();

  const Ray *ray_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray &ray = ray_ptr[i];
    const Float denom = dot(ray.m_dir, t_normal);
    Float dist = infinity<Float>();
    if(denom > 1e-6)
    {
      Vec<Float,3> p = t_point - ray.m_orig;
      const Float t = dot(p, t_normal) / denom;
      if(t > 0)
      {
        dist = t;
      }
    }

    Vec<Float,3> sample = ray.m_dir * dist + ray.m_orig;

    points_ptr[i] = sample;

  });

  return points;
}

}

Slice::Slice()
  : m_color_table("cool2warm")
{
  m_point[0] = 0.f;
  m_point[1] = 0.f;
  m_point[2] = 0.f;

  m_normal[0] = 0.f;
  m_normal[1] = 1.f;
  m_normal[2] = 0.f;
}

template<class ElemT>
Array<Vec<float32,4>>
Slice::execute(Array<Ray> &rays,
               DataSet<ElemT> &data_set)
{
  Mesh<ElemT> mesh = data_set.get_mesh();

  assert(m_field_name != "");
  dray::Shader::set_color_table(m_color_table);

  Field<FieldOn<ElemT, 1u>> field = data_set.get_field(m_field_name);

  calc_ray_start(rays, mesh.get_bounds());

  const int32 num_elems = mesh.get_num_elem();

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());

  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  Vec<float32,4> bg_color = make_vec4f(1.f,1.f,1.f,1.f);

  array_memset_vec(color_buffer, init_color);

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  Array<int32> active_rays = active_indices(rays);

  Array<Vec<Float,3>> samples = detail::calc_sample_points(rays, m_point, m_normal);


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


  array_memset(rpoints, invalid_refpt);

  // Find elements and reference coordinates for the points.
#ifdef DRAY_STATS
  mesh.locate(active_rays, samples, rpoints, *app_stats_ptr);
#else
  mesh.locate(active_rays, samples, rpoints);
#endif
  // Retrieve shading information at those points (scalar field value, gradient).
  Array<ShadingContext> shading_ctx =
    detail::get_shading_context_slice<ElemT>(rays, field, rpoints, m_normal);

  // shade and blend sample using shading context  with color buffer
  Shader::blend_surf(color_buffer, shading_ctx);

  Shader::composite_bg(color_buffer,bg_color);

  return color_buffer;
}

void
Slice::set_field(const std::string field_name)
{
 m_field_name = field_name;
}

void
Slice::set_color_table(const ColorTable &color_table)
{
  m_color_table = color_table;
}

void
Slice::set_point(const Vec<float32,3> &point)
{
  m_point = point;
}

void
Slice::set_normal(const Vec<float32,3> &normal)
{
  m_normal = normal;
  m_normal.normalize();
}

template
Array<Vec<float32,4>>
Slice::execute<MeshElem<3u, ElemType::Quad, Order::General>>(
    Array<Ray> &rays,
    DataSet<MeshElem<3u, ElemType::Quad, Order::General>> &data_set);

}//namespace dray


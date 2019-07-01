#include <dray/filters/slice.hpp>
#include <dray/filters/internal/get_shading_context.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

template<typename T>
Array<Vec<T,3>>
calc_sample_points(Array<Ray<T>> &rays,
                   const Vec<float32,3> &point,
                   const Vec<float32,3> &normal)
{
  const int32 size = rays.size();

  Array<Vec<T,3>> points;
  points.resize(size);

  Vec<T,3> t_normal;
  t_normal[0] = normal[0];
  t_normal[1] = normal[1];
  t_normal[2] = normal[2];

  Vec<T,3> t_point;
  t_point[0] = point[0];
  t_point[1] = point[1];
  t_point[2] = point[2];

  Vec<T,3> *points_ptr = points.get_device_ptr();

  const Ray<T> *ray_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

#warning "need to flip normal if coming from the opposite side, matt tired."
    const Ray<T> &ray = ray_ptr[i];
    const T denom = dot(ray.m_dir, t_normal);
    T dist = infinity<T>();
    if(denom > 1e-6)
    {
      Vec<T,3> p = t_point - ray.m_orig;
      const T t = dot(p, t_normal) / denom;
      if(t > 0)
      {
        dist = t;
      }
    }

    Vec<T,3> sample = ray.m_dir * dist + ray.m_orig;

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

template<typename T>
Array<Vec<float32,4>>
Slice::execute(Array<Ray<T>> &rays,
                          DataSet<T> &data_set)
{
  Mesh<T,3> mesh = data_set.get_mesh();

  assert(m_field_name != "");
  dray::Shader::set_color_table(m_color_table);

  Field<T> field = data_set.get_field(m_field_name);

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

  Array<Vec<T,3>> samples = detail::calc_sample_points(rays, m_point, m_normal);

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


  array_memset(rpoints, invalid_refpt);

  // Find elements and reference coordinates for the points.
#ifdef DRAY_STATS
  mesh.locate(active_rays, samples, rpoints, *app_stats_ptr);
#else
  mesh.locate(active_rays, samples, rpoints);
#endif
  // Retrieve shading information at those points (scalar field value, gradient).
  Array<ShadingContext<T>> shading_ctx =
    internal::get_shading_context(rays, field, mesh, rpoints);

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
Slice::execute<float32>(Array<Ray<float32>> &rays,
                        DataSet<float32> &data_set);

template
Array<Vec<float32,4>>
Slice::execute<float64>(Array<Ray<float64>> &rays,
                        DataSet<float64> &data_set);

}//namespace dray


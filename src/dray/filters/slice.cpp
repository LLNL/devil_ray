#include <dray/filters/slice.hpp>
#include <dray/filters/internal/get_fragments.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

Array<RayHit>
get_hits(const Array<Ray> &rays,
         const Array<Location> &locations,
         const Array<Vec<Float,3>> &points)
{
  Array<RayHit> hits;
  hits.resize(rays.size());

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const Location *loc_ptr = locations.get_device_ptr_const();
  const Vec<Float,3> *points_ptr = points.get_device_ptr_const();
  RayHit *hit_ptr = hits.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, rays.size()), [=] DRAY_LAMBDA (int32 i)
  {
    RayHit hit;
    const Location loc = loc_ptr[i];
    const Ray ray = ray_ptr[i];
    const Vec<Float,3> point = points_ptr[i];
    hit.m_hit_idx = loc.m_cell_id;
    hit.m_ref_pt  = loc.m_ref_pt;

    if(hit.m_hit_idx > -1)
    {
      hit.m_dist = (point - ray.m_orig).magnitude();
    }

    hit_ptr[i] = hit;

  });
  return hits;
}

template <class ElemT>
Array<Fragment>
get_fragments(Array<Ray> &rays,
              Field<FieldOn<ElemT, 1u>> &field,
              Array<Location> &locations,
              Vec<float32,3> &normal)
{
  const int32 size_rays = rays.size();

  Array<Fragment> fragments;
  fragments.resize(size_rays);
  Fragment *fragment_ptr = fragments.get_device_ptr();

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<Float,3> one_two_three = {123., 123., 123.};

  const int32 size = rays.size();

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const Location *loc_ptr = locations.get_device_ptr_const();

  FieldAccess<FieldOn<ElemT, 1u>> device_field = field.access_device_field();
  #warning "unify fragment and ray hit initialization"
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    Fragment frag;
    // TODO: create struct initializers
    frag.m_normal = normal;
    frag.m_scalar= 3.14f;

    const Ray &ray = ray_ptr[i];
    const Location &loc = loc_ptr[i];

    if (loc.m_cell_id >= -1)
    {
      // Compute hit point using ray origin, direction, and distance.
      //ctx.m_hit_pt = ray.m_orig + ray.m_dir * ray.m_dist;

      // Evaluate element transformation to get scalar field value and gradient.

      const int32 el_id = loc.m_cell_id;

      Vec<Vec<Float,1>,3> field_deriv;
      Vec<Float,1> scalar;
      scalar = device_field.get_elem(el_id).eval_d(loc.m_ref_pt, field_deriv);
      frag.m_scalar = scalar[0];

      if (dot(frag.m_normal, ray.m_dir) > 0.0f)
      {
        frag.m_normal = -frag.m_normal;   //Flip back toward camera.
      }
    }

    fragment_ptr[i] = frag;

  });

  return fragments;
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
void
Slice::execute(Array<Ray> &rays,
               DataSet<ElemT> &data_set,
               Framebuffer &fb)
{
  Mesh<ElemT> mesh = data_set.get_mesh();

  assert(m_field_name != "");
  dray::Shader::set_color_table(m_color_table);

  Field<FieldOn<ElemT, 1u>> field = data_set.get_field(m_field_name);

  const int32 num_elems = mesh.get_num_elem();

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());

  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  Vec<float32,4> bg_color = make_vec4f(1.f,1.f,1.f,1.f);

  array_memset_vec(color_buffer, init_color);

  //TODO: We should only cast rays that hit the AABB defined by the intersection
  // of the plane and the AABB of the mesh
  // Initial compaction: Literally remove the rays which totally miss the mesh.
  cull_missed_rays(rays, mesh.get_bounds());
  #warning "if we want to compose filters we cannot remove rays. Make a copy"
  //calc_ray_start(rays, mesh.get_bounds());
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

  Array<Location> locations;
  locations.resize(rays.size());

  //const RefPoint<3> invalid_refpt{ -1, {-1,-1,-1} };
  //array_memset(rpoints, invalid_refpt);

  // TODO change interface to locate
  Array<int32> active = array_counting(samples.size(),0,1);
  // Find elements and reference coordinates for the points.
#ifdef DRAY_STATS
  mesh.locate(active, samples, locations, *app_stats_ptr);
#else
  mesh.locate(active, samples, locations);
#endif
  // Retrieve shading information at those points (scalar field value, gradient).
  Array<Fragment> fragments =
    detail::get_fragments<ElemT>(rays, field, locations, m_normal);

  Array<RayHit> hits = detail::get_hits(rays, locations, samples);

  // shade and blend sample using shading context  with color buffer
  ColorMap color_map;
  color_map.color_table(m_color_table);
  color_map.scalar_range(field.get_range());

  Shader::blend_surf(fb, color_map, rays, hits, fragments);
  // TODO: set depth here so filters can be composible

  // TODO: this should be up to the thing that controls filters
  fb.composite_background();
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
void
Slice::execute<MeshElem<3u, ElemType::Quad, Order::General>>(
    Array<Ray> &rays,
    DataSet<MeshElem<3u, ElemType::Quad, Order::General>> &data_set,
    Framebuffer &fb);

}//namespace dray


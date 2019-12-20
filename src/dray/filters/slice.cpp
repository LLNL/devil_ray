#include <dray/filters/slice.hpp>
#include <dray/dispatcher.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/data_logger.hpp>

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

template<typename ElementType>
Array<Fragment>
get_fragments(Array<Ray> &rays,
              Field<ElementType> &field,
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

  DeviceField<ElementType> device_field(field);
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

struct Functor
{
  Slice *m_slice;
  Array<Ray> *m_rays;
  Framebuffer *m_fb;
  Functor(Slice *slice,
          Array<Ray> *rays,
          Framebuffer *fb)
    : m_slice(slice),
      m_rays(rays),
      m_fb(fb)
  {
  }

  template<typename TopologyType, typename FieldType>
  void operator()(TopologyType &topo, FieldType &field)
  {
    m_slice->execute(topo.mesh(), field, *m_rays, *m_fb);
  }
};

void
Slice::execute(Array<Ray> &rays,
               DataSet &data_set,
               Framebuffer &fb)
{
  assert(m_field_name != "");

  TopologyBase *topo = data_set.topology();
  FieldBase *field = data_set.field(m_field_name);

  Functor func(this, &rays, &fb);
  dispatch_3d(topo, field, func);
}

template<class MeshElement, class FieldElement>
void Slice::execute(Mesh<MeshElement> &mesh,
                    Field<FieldElement> &field,
                    Array<Ray> &rays,
                    Framebuffer &fb)
{
  DRAY_LOG_OPEN("slice");

  assert(m_field_name != "");
  dray::Shader::set_color_table(m_color_table);

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

  // Find elements and reference coordinates for the points.
  Array<Location> locations = mesh.locate(samples);
  // Retrieve shading information at those points (scalar field value, gradient).
  Array<Fragment> fragments =
    detail::get_fragments(rays, field, locations, m_normal);

  Array<RayHit> hits = detail::get_hits(rays, locations, samples);

  // shade and blend sample using shading context  with color buffer
  ColorMap color_map;
  color_map.color_table(m_color_table);
  color_map.scalar_range(field.get_range());

  Shader::blend_surf(fb, color_map, rays, hits, fragments);
  // TODO: set depth here so filters can be composible

  // TODO: this should be up to the thing that controls filters
  fb.composite_background();
  DRAY_LOG_CLOSE();
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

}//namespace dray


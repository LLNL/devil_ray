// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/rendering/slice_plane.hpp>
#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/GridFunction/device_field.hpp>

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
get_fragments(Field<ElementType> &field,
              Array<RayHit> &hits,
              Vec<float32,3> normal)
{
  const int32 size = hits.size();

  Array<Fragment> fragments;
  fragments.resize(size);
  Fragment *fragment_ptr = fragments.get_device_ptr();

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<Float,3> one_two_three = {123., 123., 123.};

  const RayHit *hit_ptr = hits.get_device_ptr_const();

  DeviceField<ElementType> device_field(field);
  #warning "unify fragment and ray hit initialization"
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    Fragment frag;
    // TODO: create struct initializers
    frag.m_normal = normal;
    frag.m_scalar= 3.14f;

    const RayHit &hit = hit_ptr[i];

    if (hit.m_hit_idx > -1)
    {
      // Evaluate element transformation to get scalar field value and gradient.

      const int32 el_id = hit.m_hit_idx;

      Vec<Vec<Float,1>,3> field_deriv;
      Vec<Float,1> scalar;
      scalar = device_field.get_elem(el_id).eval_d(hit.m_ref_pt, field_deriv);
      frag.m_scalar = scalar[0];
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
      if(t > 0 && t < ray.m_far && t > ray.m_near)
      {
        dist = t;
      }
    }

    Vec<Float,3> sample = ray.m_dir * dist + ray.m_orig;

    points_ptr[i] = sample;

  });

  return points;
}

} // namespace detail

SlicePlane::SlicePlane(DataSet &data_set)
  : Traceable(data_set)
{
  m_point[0] = 0.f;
  m_point[1] = 0.f;
  m_point[2] = 0.f;

  m_normal[0] = 0.f;
  m_normal[1] = 1.f;
  m_normal[2] = 0.f;
}

SlicePlane::~SlicePlane()
{
}

struct Functor
{
  SlicePlane *m_slice;
  Array<Ray> *m_rays;
  Array<RayHit> m_hits;
  Functor(SlicePlane *slice,
          Array<Ray> *rays)
    : m_slice(slice),
      m_rays(rays)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    m_hits = m_slice->execute(topo.mesh(), *m_rays);
  }
};

Array<RayHit>
SlicePlane::nearest_hit(Array<Ray> &rays)
{
  TopologyBase *topo = m_data_set.topology();

  Functor func(this, &rays);
  dispatch_3d(topo, func);
  return func.m_hits;
}

template<class MeshElement>
Array<RayHit>
SlicePlane::execute(Mesh<MeshElement> &mesh, Array<Ray> &rays)
{
  DRAY_LOG_OPEN("slice_plane");

  Array<Vec<Float,3>> samples = detail::calc_sample_points(rays, m_point, m_normal);

  // Find elements and reference coordinates for the points.
#warning "use device mesh locate"
  Array<Location> locations = mesh.locate(samples);
  std::cout<<"Done locating\n";

  Array<RayHit> hits = detail::get_hits(rays, locations, samples);

  DRAY_LOG_CLOSE();
  return hits;
}

struct FragmentFunctor
{
  SlicePlane *m_slicer;
  Array<RayHit> *m_hits;
  Array<Fragment> m_fragments;
  FragmentFunctor(SlicePlane  *slicer,
                  Array<RayHit> *hits)
    : m_slicer(slicer),
      m_hits(hits)
  {
  }

  template<typename FieldType>
  void operator()(FieldType &field)
  {
    m_fragments = detail::get_fragments(field, *m_hits, m_slicer->normal());
  }
};

Array<Fragment>
SlicePlane::fragments(Array<RayHit> &hits)
{
  DRAY_LOG_OPEN("fragments");
  assert(m_field_name != "");

  TopologyBase *topo = m_data_set.topology();
  FieldBase *field = m_data_set.field(m_field_name);

  FragmentFunctor func(this,&hits);
  dispatch_3d(field, func);
  DRAY_LOG_CLOSE();
  return func.m_fragments;
}

void
SlicePlane::point(const Vec<float32,3> &point)
{
  m_point = point;
}

Vec<float32,3>
SlicePlane::point() const
{
  return m_point;
}

void
SlicePlane::normal(const Vec<float32,3> &normal)
{
  m_normal = normal;
  m_normal.normalize();
}

Vec<float32,3>
SlicePlane::normal() const
{
  return m_normal;
}

}//namespace dray


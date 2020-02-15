// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/rendering/volume_partials.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/dispatcher.hpp>
#include <dray/error_check.hpp>
#include <dray/device_color_map.hpp>
#include <dray/face_intersection.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/device_framebuffer.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>

namespace dray
{
namespace detail
{

template<typename MeshType, typename FieldType>
DRAY_EXEC
void scalar_gradient(const Location &loc,
                     MeshType &mesh,
                     FieldType &field,
                     Float &scalar,
                     Vec<Float,3> &gradient)
{

  // i think we need this to oreient the deriv
  Vec<Vec<Float, 3>, 3> jac_vec;
  Vec<Float, 3> world_pos = // don't need this but we need the jac
    mesh.get_elem(loc.m_cell_id).eval_d(loc.m_ref_pt, jac_vec);

  Vec<Vec<Float, 1>, 3> field_deriv;
  scalar =
    field.get_elem(loc.m_cell_id).eval_d(loc.m_ref_pt, field_deriv)[0];

  Matrix<Float, 3, 3> jacobian_matrix;
  Matrix<Float, 1, 3> gradient_ref;
  for(int32 rdim = 0; rdim < 3; ++rdim)
  {
    jacobian_matrix.set_col(rdim, jac_vec[rdim]);
    gradient_ref.set_col(rdim, field_deriv[rdim]);
  }

  bool inv_valid;
  const Matrix<Float, 3, 3> j_inv = matrix_inverse(jacobian_matrix, inv_valid);
  //TODO How to handle the case that inv_valid == false?
  const Matrix<Float, 1, 3> gradient_mat = gradient_ref * j_inv;
  gradient = gradient_mat.get_row(0);
}

DRAY_EXEC_ONLY
bool intersect_AABB(const Vec<float32,4> *bvh,
                    const int32 &currentNode,
                    const Vec<Float,3> &orig_dir,
                    const Vec<Float,3> &inv_dir,
                    const Float& closest_dist,
                    bool &hit_left,
                    bool &hit_right,
                    const Float &min_dist) //Find hit after this distance
{
  Vec<float32, 4> first4  = const_get_vec4f(&bvh[currentNode + 0]);
  Vec<float32, 4> second4 = const_get_vec4f(&bvh[currentNode + 1]);
  Vec<float32, 4> third4  = const_get_vec4f(&bvh[currentNode + 2]);
  Float xmin0 = first4[0] * inv_dir[0] - orig_dir[0];
  Float ymin0 = first4[1] * inv_dir[1] - orig_dir[1];
  Float zmin0 = first4[2] * inv_dir[2] - orig_dir[2];
  Float xmax0 = first4[3] * inv_dir[0] - orig_dir[0];
  Float ymax0 = second4[0] * inv_dir[1] - orig_dir[1];
  Float zmax0 = second4[1] * inv_dir[2] - orig_dir[2];
  Float min0 = fmaxf(
    fmaxf(fmaxf(fminf(ymin0, ymax0), fminf(xmin0, xmax0)), fminf(zmin0, zmax0)),
    min_dist);
  Float max0 = fminf(
    fminf(fminf(fmaxf(ymin0, ymax0), fmaxf(xmin0, xmax0)), fmaxf(zmin0, zmax0)),
    closest_dist);
  hit_left = (max0 >= min0);

  Float xmin1 = second4[2] * inv_dir[0] - orig_dir[0];
  Float ymin1 = second4[3] * inv_dir[1] - orig_dir[1];
  Float zmin1 = third4[0] * inv_dir[2] - orig_dir[2];
  Float xmax1 = third4[1] * inv_dir[0] - orig_dir[0];
  Float ymax1 = third4[2] * inv_dir[1] - orig_dir[1];
  Float zmax1 = third4[3] * inv_dir[2] - orig_dir[2];

  Float min1 = fmaxf(
    fmaxf(fmaxf(fminf(ymin1, ymax1), fminf(xmin1, xmax1)), fminf(zmin1, zmax1)),
    min_dist);
  Float max1 = fminf(
    fminf(fminf(fmaxf(ymin1, ymax1), fmaxf(xmin1, xmax1)), fmaxf(zmin1, zmax1)),
    closest_dist);
  hit_right = (max1 >= min1);

  return (min0 > min1);
}

template<class ElemT>
struct FaceIntersector
{
  DeviceMesh<ElemT> m_device_mesh;

  FaceIntersector(DeviceMesh<ElemT> &device_mesh)
   : m_device_mesh(device_mesh)
  {}

  DRAY_EXEC RayHit intersect_face(const Ray &ray,
                                  const int32 &el_idx,
                                  const AABB<2> &ref_box,
                                  stats::Stats &mstat) const
  {
    const bool use_init_guess = true;
    RayHit hit;
    hit.m_hit_idx = -1;

    mstat.acc_candidates(1);
    Vec<Float,2> ref = ref_box.template center<Float> ();
    hit.m_dist = ray.m_near;

    bool inside = Intersector_RayFace<ElemT>::intersect_local (mstat,
                                              m_device_mesh.get_elem(el_idx),
                                              ray,
                                              ref,// initial ref guess
                                              hit.m_dist,  // initial ray guess
                                              use_init_guess);
    if(inside)
    {
      hit.m_hit_idx = el_idx;
      hit.m_ref_pt[0] = ref[0];
      hit.m_ref_pt[1] = ref[1];
    }
    return hit;
  }

};

template <int MAX_DEPTH>
struct DistQ
{
  int32 m_nodes[MAX_DEPTH];
  Float m_distances[MAX_DEPTH];
  int32 m_index;

  DRAY_EXEC void init()
  {
    m_index = -1;
  }

  DRAY_EXEC bool empty()
  {
    return m_index == -1;
  }

  DRAY_EXEC Float peek()
  {
    // could return inf if empty
    return m_distances[index];
  }

  DRAY_EXEC Float pop(int32 &node)
  {
    node = m_nodes[m_index];
    Float dist = m_distances[m_index];
    m_index--;
    return dist;
  }

  DRAY_EXEC void insert(const int32 &node, const Float &dist)
  {
    // im sure there are better ways to do this, but i want
    // simple code -ml
    m_index += 1;
    m_nodes[m_index] = node;
    m_distances[m_index] = dist;

    for(int32 i = m_index; i > 0; --i)
    {
      if(m_distances[i] > m_distances[i-1])
      {
        Float tmp_f = m_distances[i];
        int32 tmp_i = m_nodes[i];

        m_distances[i] = m_distances[i-1];
        m_nodes[i] = m_nodes[i-1];

        m_distances[i-1] = tmp_f;
        m_nodes[i-1] = m_nodes[i-1];
      }
      else break;
    }
  }
};

template <typename ElemT>
Array<RayHit> intersect_faces(Array<Ray> rays, Mesh<ElemT> &mesh)
{
  const int32 size = rays.size();
  Array<RayHit> hits;
  hits.resize(size);

  const BVH bvh = mesh.get_bvh();

  const Ray *ray_ptr = rays.get_device_ptr_const();
  RayHit *hit_ptr = hits.get_device_ptr();

  const int32 *leaf_ptr = bvh.m_leaf_nodes.get_device_ptr_const();
  const int32 *aabb_ids_ptr = bvh.m_aabb_ids.get_device_ptr_const();
  const Vec<float32, 4> *inner_ptr = bvh.m_inner_nodes.get_device_ptr_const();
  const AABB<2> *ref_aabb_ptr = mesh.get_ref_aabbs().get_device_ptr_const();

  DeviceMesh<ElemT> device_mesh(mesh);
  FaceIntersector<ElemT> intersector(device_mesh);

  Array<stats::Stats> mstats;
  mstats.resize(size);
  stats::Stats *mstats_ptr = mstats.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray ray = ray_ptr[i];

    RayHit hit;
    hit.m_hit_idx = -1;

    stats::Stats mstat;
    mstat.construct();

    Float closest_dist = ray.m_far;
    Float min_dist = ray.m_near;
    const Vec<Float,3> dir = ray.m_dir;
    Vec<Float,3> inv_dir;
    inv_dir[0] = rcp_safe(dir[0]);
    inv_dir[1] = rcp_safe(dir[1]);
    inv_dir[2] = rcp_safe(dir[2]);

    int32 current_node;
    int32 todo[64];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;

    const Vec<Float,3> orig = ray.m_orig;

    Vec<Float,3> orig_dir;
    orig_dir[0] = orig[0] * inv_dir[0];
    orig_dir[1] = orig[1] * inv_dir[1];
    orig_dir[2] = orig[2] * inv_dir[2];

    while (current_node != barrier)
    {
      if (current_node > -1)
      {
        bool hit_left, hit_right;
        bool right_closer = intersect_AABB(inner_ptr,
                                           current_node,
                                           orig_dir,
                                           inv_dir,
                                           closest_dist,
                                           hit_left,
                                           hit_right,
                                           min_dist);

        if (!hit_left && !hit_right)
        {
          current_node = todo[stackptr];
          stackptr--;
        }
        else
        {
          Vec<float32, 4> children = const_get_vec4f(&inner_ptr[current_node + 3]);
          int32 l_child;
          constexpr int32 isize = sizeof(int32);
          memcpy(&l_child, &children[0], isize);
          int32 r_child;
          memcpy(&r_child, &children[1], isize);
          current_node = (hit_left) ? l_child : r_child;

          if (hit_left && hit_right)
          {
            if (right_closer)
            {
              current_node = r_child;
              stackptr++;
              todo[stackptr] = l_child;
            }
            else
            {
              stackptr++;
              todo[stackptr] = r_child;
            }
          }
        }
      } // if inner node

      if (current_node < 0 && current_node != barrier) //check register usage
      {
        current_node = -current_node - 1; //swap the neg address

        int32 el_idx = leaf_ptr[current_node];
        const AABB<2> ref_box = ref_aabb_ptr[aabb_ids_ptr[current_node]];

        RayHit el_hit = intersector.intersect_face(ray, el_idx, ref_box, mstat);

        if(el_hit.m_hit_idx != -1 && el_hit.m_dist < closest_dist && el_hit.m_dist > min_dist)
        {
          hit = el_hit;
          closest_dist = hit.m_dist;
          mstat.found();
        }

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } //while

    mstats_ptr[i] = mstat;
    hit_ptr[i] = hit;

  });
  DRAY_ERROR_CHECK();

  stats::StatStore::add_ray_stats(rays, mstats);
  return hits;
}

template<typename MeshElement, typename FieldElement>
void volume_integrate(Mesh<MeshElement> &mesh,
                      Field<FieldElement> &field,
                      Array<Ray> &rays,
                      Array<PointLight> &lights,
                      const int32 samples,
                      ColorMap &color_map)
{
  DRAY_LOG_OPEN("volume");

  assert(m_field_name != "");

#if 0
  constexpr float32 correction_scalar = 10.f;
  float32 ratio = correction_scalar / samples;
  ColorMap corrected = color_map;
  ColorTable table = corrected.color_table();
  corrected.color_table(table.correct_opacity(ratio));

  dray::AABB<> bounds = mesh.get_bounds();
  dray::float32 mag = (bounds.max() - bounds.min()).magnitude();
  const float32 sample_dist = mag / dray::float32(samples);

  const int32 num_elems = mesh.get_num_elem();

  DRAY_LOG_ENTRY("samples", samples);
  DRAY_LOG_ENTRY("sample_distance", sample_dist);
  DRAY_LOG_ENTRY("cells", num_elems);
  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  Array<Ray> active_rays = remove_missed_rays(rays, mesh.get_bounds());

  const int32 ray_size = active_rays.size();
  const Ray *rays_ptr = active_rays.get_device_ptr_const();

  // complicated device stuff
  DeviceMesh<MeshElement> device_mesh(mesh);
  DeviceFramebuffer d_framebuffer(fb);
  DeviceField<FieldElement> device_field(field);

  DeviceColorMap d_color_map(corrected);

  const PointLight *light_ptr = lights.get_device_ptr_const();
  const int32 num_lights = lights.size();

  // TODO: somehow load balance based on far - near

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 i)
  {
    const Ray ray = rays_ptr[i];
    // advance the ray one step
    Float distance = ray.m_near + sample_dist;
    Vec4f color = {0.f, 0.f, 0.f, 0.f};;
    while(distance < ray.m_far)
    {
      //Vec<Float,3> point = ray.m_orig + ray.m_dir * distance;
      Vec<Float,3> point = ray.m_orig + distance * ray.m_dir;
      Location loc = device_mesh.locate(point);
      if(loc.m_cell_id != -1)
      {
        Vec<Float,3> gradient;
        Float scalar;
        detail::scalar_gradient(loc, device_mesh, device_field, scalar, gradient);
        Vec4f sample_color = d_color_map.color(scalar);

        //composite
        blend(color, sample_color);
        if(color[3] > 0.95f)
        {
          // terminate
          distance = ray.m_far;
        }
      }

      distance += sample_dist;
    }
    Vec4f back_color = d_framebuffer.m_colors[ray.m_pixel_id];
    blend(color, back_color);
    d_framebuffer.m_colors[ray.m_pixel_id] = color;
    // should this be first valid sample or even set this?
    //d_framebuffer.m_depths[pid] = hit.m_dist;

  });
  DRAY_ERROR_CHECK();

#endif
  DRAY_LOG_CLOSE();
}

// ------------------------------------------------------------------------
struct IntegratePartialsFunctor
{
  Array<Ray> *m_rays;
  Array<PointLight> m_lights;
  ColorMap &m_color_map;
  Float m_samples;
  IntegratePartialsFunctor(Array<Ray> *rays,
                           Array<PointLight> &lights,
                           ColorMap &color_map,
                           Float samples)
    :
      m_rays(rays),
      m_lights(lights),
      m_color_map(color_map),
      m_samples(samples)

  {
  }

  template<typename TopologyType, typename FieldType>
  void operator()(TopologyType &topo, FieldType &field)
  {
    detail::volume_integrate(topo.mesh(),
                             field,
                             *m_rays,
                             m_lights,
                             m_samples,
                             m_color_map);
  }
};

struct SegmentFunctor
{
  Array<Ray> *m_rays;
  SegmentFunctor(Array<Ray> *rays)
    : m_rays(rays)

  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    Timer timer;
    Array<RayHit> hits = intersect_faces(*m_rays, topo.mesh());
    float time = timer.elapsed();
    std::cout<<"time "<<time<<"\n";
  }
};
} // namespace detail

// ------------------------------------------------------------------------
VolumePartial::VolumePartial(DataSet &data_set)
  : m_data_set(data_set),
    m_samples(100)
{
  // add some default alpha
  ColorTable table = m_color_map.color_table();
  table.add_alpha(0.1000, .0f);
  table.add_alpha(1.0000, .7f);
  m_color_map.color_table(table);

  dray::MeshBoundary boundary;
  m_boundary = boundary.execute(data_set);
}

// ------------------------------------------------------------------------
VolumePartial::~VolumePartial()
{
}

// ------------------------------------------------------------------------
void
VolumePartial::input(DataSet &data_set)
{
  m_data_set = data_set;

  dray::MeshBoundary boundary;
  m_boundary = boundary.execute(data_set);
}

// ------------------------------------------------------------------------

void
VolumePartial::field(const std::string field)
{
  m_field = field;
}

// ------------------------------------------------------------------------

void
VolumePartial::integrate(Array<Ray> &rays, Array<PointLight> &lights)
{
  if(m_field == "")
  {
    DRAY_ERROR("Field never set");
  }

  if(!m_color_map.range_set())
  {
    std::vector<Range> ranges  = m_data_set.field(m_field)->range();
    if(ranges.size() != 1)
    {
      DRAY_ERROR("Expected 1 range component, got "<<ranges.size());
    }
    m_color_map.scalar_range(ranges[0]);
  }

  TopologyBase *boundary_topo = m_boundary.topology();

  TopologyBase *topo = m_data_set.topology();
  FieldBase *field = m_data_set.field(m_field);

  detail::SegmentFunctor seg_func(&rays);
  dispatch_2d(boundary_topo,seg_func);

  detail::IntegratePartialsFunctor func( &rays, lights, m_color_map, m_samples);
  dispatch_3d(topo, field, func);
}
// ------------------------------------------------------------------------

void VolumePartial::samples(int32 num_samples)
{
  m_samples = num_samples;
}
// ------------------------------------------------------------------------
//
//Array<RayHit> VolumePartial::nearest_hit(Array<Ray> &rays)
//{
//  // this is a placeholder
//  // Possible implementations include hitting the bounding box
//  // or actually hitting the external faces. When we support mpi
//  // volume rendering, we will need to extract partial composites
//  // since there is no promise, ever, about domain decomposition
//  Array<RayHit> hits;
//  DRAY_ERROR("not implemented");
//  return hits;
//}
} // namespace dray

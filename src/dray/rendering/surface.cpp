// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/rendering/surface.hpp>
#include <dray/rendering/colors.hpp>

#include <dray/GridFunction/device_mesh.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/array_utils.hpp>
#include <dray/dispatcher.hpp>
#include <dray/ref_point.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/face_intersection.hpp>
#include <dray/utils/data_logger.hpp>


namespace dray
{
namespace detail
{

class ShadeMeshLines
{
  protected:
  Vec4f u_edge_color;
  Vec4f u_face_color;
  float32 u_edge_radius_rcp;
  int32 u_grid_res;

  public:
  void set_uniforms (Vec4f edge_color, Vec4f face_color, float32 edge_radius, int32 grid_res = 1)
  {
    u_edge_color = edge_color;
    u_face_color = face_color;
    u_edge_radius_rcp = (edge_radius > 0.0 ? 1.0 / edge_radius : 0.05) / grid_res;
    u_grid_res = grid_res;
  }

  DRAY_EXEC Vec4f operator() (const Vec<Float, 2> &rcoords) const
  {
    // Get distance to nearest edge.
    float32 edge_dist = 0.0;
    {
      Vec<Float, 2> prcoords = rcoords;
      prcoords[0] = u_grid_res * prcoords[0];
      prcoords[0] -= floor (prcoords[0]);
      prcoords[1] = u_grid_res * prcoords[1];
      prcoords[1] -= floor (prcoords[1]);

      float32 d0 =
      (prcoords[0] < 0.0 ? 0.0 : prcoords[0] > 1.0 ? 0.0 : 0.5 - fabs (prcoords[0] - 0.5));
      float32 d1 =
      (prcoords[1] < 0.0 ? 0.0 : prcoords[1] > 1.0 ? 0.0 : 0.5 - fabs (prcoords[1] - 0.5));

      float32 min2 = (d0 < d1 ? d0 : d1);
      edge_dist = min2;
    }
    edge_dist *= u_edge_radius_rcp; // Normalized distance from nearest edge.
    // edge_dist is nonnegative.

    const float32 x = min (edge_dist, 1.0f);

    // Cubic smooth interpolation.
    float32 w = (2.0 * x - 3.0) * x * x + 1.0;
    Vec4f frag_color = u_edge_color * w + u_face_color * (1.0 - w);
    return frag_color;
  }

  DRAY_EXEC Vec4f operator() (const Vec<Float, 3> &rcoords) const
  {
    // Since it is assumed one of the coordinates is 0.0 or 1.0 (we have a face
    // point), we want to measure the second-nearest-to-edge distance.

    float32 edge_dist = 0.0;
    {
      Vec<Float, 3> prcoords = rcoords;
      prcoords[0] = u_grid_res * prcoords[0];
      prcoords[0] -= floor (prcoords[0]);
      prcoords[1] = u_grid_res * prcoords[1];
      prcoords[1] -= floor (prcoords[1]);
      prcoords[2] = u_grid_res * prcoords[2];
      prcoords[2] -= floor (prcoords[2]);

      float32 d0 =
      (prcoords[0] < 0.0 ? 0.0 : prcoords[0] > 1.0 ? 0.0 : 0.5 - fabs (prcoords[0] - 0.5));
      float32 d1 =
      (prcoords[1] < 0.0 ? 0.0 : prcoords[1] > 1.0 ? 0.0 : 0.5 - fabs (prcoords[1] - 0.5));
      float32 d2 =
      (prcoords[2] < 0.0 ? 0.0 : prcoords[2] > 1.0 ? 0.0 : 0.5 - fabs (prcoords[2] - 0.5));

      float32 min2 = (d0 < d1 ? d0 : d1);
      float32 max2 = (d0 < d1 ? d1 : d0);
      // Now three cases: d2 < min2 <= max2;   min2 <= d2 <= max2;   min2 <= max2 < d2;
      edge_dist = (d2 < min2 ? min2 : max2 < d2 ? max2 : d2);
    }

    edge_dist *= u_edge_radius_rcp; // Normalized distance from nearest edge.
    // edge_dist is nonnegative.

    const float32 x = min (edge_dist, 1.0f);

    // Cubic smooth interpolation.
    float32 w = (2.0 * x - 3.0) * x * x + 1.0;

    float32 alpha =  u_edge_color[3] * w;
    Vec4f frag_color = u_edge_color;
    frag_color[3] = alpha;
    return frag_color;
  }
};

//
// intersect_AABB()
//
// Copied verbatim from triangle_mesh.cpp
//
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

struct HasCandidate
{
  int32 m_max_candidates;
  const int32 *m_candidates_ptr;
  DRAY_EXEC bool operator() (int32 ii) const
  {
    return (m_candidates_ptr[ii * m_max_candidates] > -1);
  }
};

template<typename MeshElem>
Array<RayHit>
surface_execute(Mesh<MeshElem> &mesh,
                Array<Ray> &rays)
{
  DRAY_LOG_OPEN("surface_intersection");

  Array<RayHit> hits = intersect_faces(rays, mesh);

  DRAY_LOG_CLOSE();
  return hits;
}

struct SurfaceFunctor
{
  Array<Ray> *m_rays;
  Array<RayHit> m_hits;

  SurfaceFunctor(Array<Ray> *rays)
    : m_rays(rays)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    m_hits = surface_execute(topo.mesh(), *m_rays);
  }
};


}  // namespace detail


Surface::Surface(DataSet &dataset)
  : Traceable(dataset),
    m_draw_mesh(false),
    m_line_thickness(0.05f),
    m_sub_res(1.f)
{
  m_line_color = make_vec4f(0.f, 0.f, 0.f, 1.f);
}

Surface::~Surface()
{
}

Array<RayHit>
Surface::nearest_hit(Array<Ray> &rays)
{
  TopologyBase *topo = m_data_set.topology();

  detail::SurfaceFunctor func(&rays);
  dispatch_2d(topo, func);
  return func.m_hits;
}

void Surface::draw_mesh(bool on)
{
  m_draw_mesh = on;
}

void Surface::line_thickness(const float32 thickness)
{
  if(thickness <= 0.f)
  {
    DRAY_ERROR("Cannot have 0 thickness");
  }

  m_line_thickness = thickness;
}

void Surface::shade(const Array<Ray> &rays,
                    const Array<RayHit> &hits,
                    const Array<Fragment> &fragments,
                    const Array<PointLight> &lights,
                    Framebuffer &framebuffer)
{
  Traceable::shade(rays, hits, fragments, lights, framebuffer);

  if(m_draw_mesh)
  {
    DeviceFramebuffer d_framebuffer(framebuffer);
    const RayHit *hit_ptr = hits.get_device_ptr_const();
    const Ray *rays_ptr = rays.get_device_ptr_const();

    // Initialize fragment shader.
    detail::ShadeMeshLines shader;
    // todo: get from framebuffer
    const Vec<float32,4> face_color = make_vec4f(0.f, 0.f, 0.f, 0.f);
    shader.set_uniforms(m_line_color, face_color, m_line_thickness, m_sub_res);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, hits.size()), [=] DRAY_LAMBDA (int32 ii)
    {
      const RayHit &hit = hit_ptr[ii];
      if (hit.m_hit_idx != -1)
      {
        Color current = d_framebuffer.m_colors[rays_ptr[ii].m_pixel_id];
        Vec<float32,4> pixel_color = shader(hit.m_ref_pt);
        blend(pixel_color, current);
        d_framebuffer.m_colors[rays_ptr[ii].m_pixel_id] = pixel_color;
      }
    });
    DRAY_ERROR_CHECK();
  }

}

void Surface::line_color(const Vec<float32,4> &color)
{
  m_line_color = color;
}
};//naemespace dray

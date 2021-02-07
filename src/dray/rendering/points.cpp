// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/rendering/points.hpp>
#include <dray/rendering/colors.hpp>

#include <dray/point_topology.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/array_utils.hpp>
#include <dray/dispatcher.hpp>
#include <dray/ref_point.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/rendering/low_order_intersectors.hpp>
#include <dray/face_intersection.hpp>
#include <dray/utils/data_logger.hpp>


namespace dray
{
namespace detail
{

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

Array<RayHit> intersect_points(Array<Ray> rays, PointTopology &points)
{
  const int32 size = rays.size();
  Array<RayHit> hits;
  hits.resize(size);

  const Ray *ray_ptr = rays.get_device_ptr_const();
  RayHit *hit_ptr = hits.get_device_ptr();

  const BVH bvh = points.bvh();
  const int32 *leaf_ptr = bvh.m_leaf_nodes.get_device_ptr_const();
  const int32 *aabb_ids_ptr = bvh.m_aabb_ids.get_device_ptr_const();
  const Vec<float32, 4> *inner_ptr = bvh.m_inner_nodes.get_device_ptr_const();

  const Vec<Float,3> *points_ptr = points.points().get_device_ptr_const();
  const Float *radii_ptr = points.radii().get_device_ptr_const();

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


        Vec<Float,3> center = points_ptr[el_idx];
        Float radius = radii_ptr[el_idx];
        Float distance = intersect_sphere(center, radius, ray.m_orig, ray.m_dir);

        RayHit el_hit;
        el_hit.m_hit_idx = -1;

        if(distance != infinity<Float>())
        {
          el_hit.m_hit_idx = el_idx;
          el_hit.m_dist = distance;
          // COMPLETE HACK since in the fragements call we don't
          // have enough information to recreate the normal
          Vec<Float,3> hit = ray.m_orig + ray.m_dir * distance;
          Vec<Float,3> normal = hit - center;
          normal.normalize();
          el_hit.m_ref_pt = normal;
        }

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

}  // namespace detail


Points::Points(Collection &collection)
  : Traceable(collection)
{
  m_color = make_vec4f(1.f, 1.f, 1.f, 1.f);
}

Points::~Points()
{
}

Array<RayHit>
Points::nearest_hit(Array<Ray> &rays)
{
  DataSet data_set = m_collection.domain(m_active_domain);
  TopologyBase *topo = data_set.topology();
  PointTopology *point_topo = static_cast<PointTopology*>(topo);

  if(point_topo == nullptr)
  {
    DRAY_ERROR("Points needs a point topology");
  }

  DRAY_LOG_OPEN("point_intersection");
  Array<RayHit> hits = detail::intersect_points(rays, *point_topo);
  DRAY_LOG_CLOSE();

  return hits;
}

Array<Fragment>
Points::fragments(Array<RayHit> &hits)
{
  DRAY_LOG_OPEN("fragments");
  //if(m_field_name == "")
  //{
  //  DRAY_ERROR("Field name never set");
  //}

  DataSet data_set = m_collection.domain(m_active_domain);

  TopologyBase *topo = data_set.topology();

  PointTopology *point_topo = static_cast<PointTopology*>(topo);

  if(point_topo == nullptr)
  {
    DRAY_ERROR("Points needs a point topology");
  }

  const int32 size = hits.size();

  Array<Fragment> fragments;
  fragments.resize(size);
  Fragment *fragments_ptr = fragments.get_device_ptr();

  const RayHit *hit_ptr = hits.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Fragment frag;
    frag.m_scalar = 0.f;
    // hack we don't have enough info to recreate the hit point
    frag.m_normal = hit_ptr[i].m_ref_pt;
    fragments_ptr[i] = frag;

  });
  DRAY_ERROR_CHECK();

  DRAY_LOG_CLOSE();
  return fragments;
}

void Points::shade(const Array<Ray> &rays,
                   const Array<RayHit> &hits,
                   const Array<Fragment> &fragments,
                   Framebuffer &framebuffer)
{

}
void Points::colors(const Array<Ray> &rays,
                    const Array<RayHit> &hits,
                    const Array<Fragment> &fragments,
                    Array<Vec<float32,4>> &colors)
{

  colors.resize(rays.size());
  Vec<float32,4> *color_ptr = colors.get_device_ptr();
  //DeviceColorMap d_color_map (m_color_map);
  const Vec<float32,4> color = m_color;

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, hits.size ()), [=] DRAY_LAMBDA (int32 ii)
  {
    color_ptr[ii] = color;
  });

}

void Points::shade(const Array<Ray> &rays,
                   const Array<RayHit> &hits,
                   const Array<Fragment> &fragments,
                   const Array<PointLight> &lights,
                   Framebuffer &framebuffer)
{

}


void Points::constant_color(const Vec<float32,4> &color)
{
  m_color = color;
}
};//naemespace dray

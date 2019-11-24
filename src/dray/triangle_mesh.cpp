// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/triangle_mesh.hpp>

#include <dray/array_utils.hpp>
#include <dray/intersection_context.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/policies.hpp>
#include <dray/triangle_intersection.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{


Array<AABB<>> get_tri_aabbs (Array<float32> &coords, Array<int32> indices)
{
  Array<AABB<>> aabbs;

  assert (indices.size () % 3 == 0);
  const int32 num_tris = indices.size () / 3;

  aabbs.resize (num_tris);

  const int32 *indices_ptr = indices.get_device_ptr_const ();
  const float32 *coords_ptr = coords.get_device_ptr_const ();
  AABB<> *aabb_ptr = aabbs.get_device_ptr ();

  std::cout << "number of triangles " << num_tris << "\n";
  std::cout << "coords " << coords.size () << "\n";

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_tris), [=] DRAY_LAMBDA (int32 tri) {
    AABB<> aabb;

    const int32 i_offset = tri * 3;

    for (int32 i = 0; i < 3; ++i)
    {
      const int32 vertex_id = indices_ptr[i_offset + i];
      const int32 v_offset = vertex_id * 3;
      Vec3f vertex;

      for (int32 v = 0; v < 3; ++v)
      {
        vertex[v] = coords_ptr[v_offset + v];
      }
      aabb.include (vertex);
    }
    aabb_ptr[tri] = aabb;
  });


  return aabbs;
}

} // namespace detail


TriangleMesh::TriangleMesh (Array<float32> &coords, Array<int32> &indices)
: m_coords (coords), m_indices (indices)
{
  Array<AABB<>> aabbs = detail::get_tri_aabbs (m_coords, indices);

  LinearBVHBuilder builder;
  m_bvh = builder.construct (aabbs);
}

TriangleMesh::TriangleMesh ()
{
}

TriangleMesh::~TriangleMesh ()
{
}

Array<float32> &TriangleMesh::get_coords ()
{
  return m_coords;
}

Array<int32> &TriangleMesh::get_indices ()
{
  return m_indices;
}

AABB<> TriangleMesh::get_bounds ()
{
  return m_bvh.m_bounds;
}

template <typename T>
DRAY_EXEC_ONLY bool intersect_AABB (const Vec<float32, 4> *bvh,
                                    const int32 &currentNode,
                                    const Vec<T, 3> &orig_dir,
                                    const Vec<T, 3> &inv_dir,
                                    const T &closest_dist,
                                    bool &hit_left,
                                    bool &hit_right,
                                    const T &min_dist) // Find hit after this distance
{
  Vec<float32, 4> first4 = const_get_vec4f (&bvh[currentNode + 0]);
  Vec<float32, 4> second4 = const_get_vec4f (&bvh[currentNode + 1]);
  Vec<float32, 4> third4 = const_get_vec4f (&bvh[currentNode + 2]);
  T xmin0 = first4[0] * inv_dir[0] - orig_dir[0];
  T ymin0 = first4[1] * inv_dir[1] - orig_dir[1];
  T zmin0 = first4[2] * inv_dir[2] - orig_dir[2];
  T xmax0 = first4[3] * inv_dir[0] - orig_dir[0];
  T ymax0 = second4[0] * inv_dir[1] - orig_dir[1];
  T zmax0 = second4[1] * inv_dir[2] - orig_dir[2];
  T min0 =
  fmaxf (fmaxf (fmaxf (fminf (ymin0, ymax0), fminf (xmin0, xmax0)), fminf (zmin0, zmax0)),
         min_dist);
  T max0 =
  fminf (fminf (fminf (fmaxf (ymin0, ymax0), fmaxf (xmin0, xmax0)), fmaxf (zmin0, zmax0)),
         closest_dist);
  hit_left = (max0 >= min0);

  T xmin1 = second4[2] * inv_dir[0] - orig_dir[0];
  T ymin1 = second4[3] * inv_dir[1] - orig_dir[1];
  T zmin1 = third4[0] * inv_dir[2] - orig_dir[2];
  T xmax1 = third4[1] * inv_dir[0] - orig_dir[0];
  T ymax1 = third4[2] * inv_dir[1] - orig_dir[1];
  T zmax1 = third4[3] * inv_dir[2] - orig_dir[2];

  T min1 =
  fmaxf (fmaxf (fmaxf (fminf (ymin1, ymax1), fminf (xmin1, xmax1)), fminf (zmin1, zmax1)),
         min_dist);
  T max1 =
  fminf (fminf (fminf (fmaxf (ymin1, ymax1), fmaxf (xmin1, xmax1)), fmaxf (zmin1, zmax1)),
         closest_dist);
  hit_right = (max1 >= min1);

  return (min0 > min1);
}

Array<RayHit> TriangleMesh::intersect (const Array<Ray> &rays)
{
  const float32 *coords_ptr = m_coords.get_device_ptr_const ();
  const int32 *indices_ptr = m_indices.get_device_ptr_const ();
  const int32 *leaf_ptr = m_bvh.m_leaf_nodes.get_device_ptr_const ();
  const Vec<float32, 4> *inner_ptr = m_bvh.m_inner_nodes.get_device_ptr_const ();

  const Ray *ray_ptr = rays.get_device_ptr_const ();

  const int32 size = rays.size ();

  Array<RayHit> hits;
  hits.resize (size);

  RayHit *hit_ptr = hits.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    Ray ray = ray_ptr[i];
    RayHit hit;

    Float closest_dist = ray.m_far;
    Float min_dist = ray.m_near;
    int32 hit_idx = -1;
    const Vec<Float, 3> dir = ray.m_dir;
    Vec<Float, 3> inv_dir;
    inv_dir[0] = rcp_safe (dir[0]);
    inv_dir[1] = rcp_safe (dir[1]);
    inv_dir[2] = rcp_safe (dir[2]);

    int32 current_node;
    int32 todo[64];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;

    Vec<Float, 3> orig_dir;
    orig_dir[0] = ray.m_orig[0] * inv_dir[0];
    orig_dir[1] = ray.m_orig[1] * inv_dir[1];
    orig_dir[2] = ray.m_orig[2] * inv_dir[2];

    while (current_node != barrier)
    {
      if (current_node > -1)
      {
        bool hit_left, hit_right;
        bool right_closer = intersect_AABB (inner_ptr, current_node, orig_dir, inv_dir,
                                            closest_dist, hit_left, hit_right, min_dist);

        if (!hit_left && !hit_right)
        {
          current_node = todo[stackptr];
          stackptr--;
        }
        else
        {
          Vec<float32, 4> children = const_get_vec4f (&inner_ptr[current_node + 3]);
          int32 l_child;
          constexpr int32 isize = sizeof (int32);
          memcpy (&l_child, &children[0], isize);
          int32 r_child;
          memcpy (&r_child, &children[1], isize);
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

      if (current_node < 0 && current_node != barrier) // check register usage
      {
        current_node = -current_node - 1; // swap the neg address
        Float minU, minV;
        // Moller leaf_intersector;
        TriLeafIntersector<Moller> leaf_intersector;
        leaf_intersector.intersect_leaf (current_node, ray.m_orig, dir, hit_idx,
                                         minU, minV, closest_dist, min_dist,
                                         indices_ptr, coords_ptr, leaf_ptr);

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } // while

    if (hit_idx != -1)
    {
      hit.m_dist = closest_dist;
    }

    hit.m_hit_idx = hit_idx;
    hit_ptr[i] = hit;
  });
  return hits;
}


Array<IntersectionContext>
TriangleMesh::get_intersection_context (const Array<Ray> &rays, const Array<RayHit> &hits)
{
  const int32 size = rays.size ();

  Array<IntersectionContext> intersection_ctx;
  intersection_ctx.resize (size);

  // Device pointers for output
  IntersectionContext *ctx_ptr = intersection_ctx.get_device_ptr ();

  // Read-only device pointers for input fields.
  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const RayHit *hit_ptr = hits.get_device_ptr_const ();

  // Read-only device pointers for mesh object member fields.
  const float32 *m_coords_ptr = m_coords.get_device_ptr_const ();
  const int32 *m_indices_ptr = m_indices.get_device_ptr_const ();

  RAJA::View<const float32, RAJA::Layout<2>> coords (m_coords_ptr,
                                                     m_coords.size () / 3, 3);
  RAJA::View<const int32, RAJA::Layout<2>> indices (m_indices_ptr,
                                                    m_indices.size () / 3, 3);

  // Iterate over all rays.
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ray_idx) {
    const Ray &ray = ray_ptr[ray_idx];
    const RayHit &hit = hit_ptr[ray_idx];
    IntersectionContext ctx;

    ctx.m_pixel_id = ray.m_pixel_id;
    ctx.m_ray_dir = ray.m_dir;

    if (hit.m_hit_idx == -1)
    {
      // There is no intersection.
      ctx.m_is_valid = 0;
    }
    else
    {
      // There is an intersection.
      ctx.m_is_valid = 1;

      // Calculate the hit point by projecting the ray.
      ctx.m_hit_pt = ray.m_orig + ray.m_dir * hit.m_dist;

      // Get the triangle vertex coordinates (to later calculate surface normal).
      Vec<Float, 3> v[3];

      // Using raw int32 and saving intermediate indices...
      /// const int32 i_offset = in_hit_idx_ptr[ray_idx] * 3;
      /// for(int32 i = 0; i < 3; ++i)
      /// {
      ///   const int32 vertex_id = m_indices_ptr[i_offset + i];
      ///   const int32 v_offset = vertex_id * 3;

      ///   for(int32 vi = 0; vi < 3; ++vi)
      ///   {
      ///     v[i][vi] = (T) m_coords_ptr[v_offset + vi];
      ///   }
      /// }

      // Using RAJA "Views"...
      for (int32 i = 0; i < 3; ++i)
        for (int32 vi = 0; vi < 3; ++vi)
          v[i][vi] = (Float)coords (indices (hit.m_hit_idx, i), vi);

      // Now calculate the surface normal (facing the source of the ray).
      ctx.m_normal = cross (v[1] - v[0], v[2] - v[0]);
      ctx.m_normal.normalize ();
      if (dot (ctx.m_normal, ray.m_dir) > 0.0f)
      {
        ctx.m_normal = -ctx.m_normal;
      }
    }

    ctx_ptr[ray_idx] = ctx;
  });

  return intersection_ctx;
}

} // namespace dray

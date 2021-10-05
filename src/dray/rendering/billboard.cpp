// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/billboard.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/rendering/font_factory.hpp>
#include <dray/array_utils.hpp>
#include <dray/matrix.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/rendering/screen_text_annotator.hpp>

namespace dray
{

namespace detail
{

AABB<3> bound_sphere(const Vec<float32,3> &center, const float32 radius)
{
  AABB<3> res;
  Vec<float32,3> temp;

  temp[0] = radius;
  temp[1] = 0.f;
  temp[2] = 0.f;

  res.include(center + temp);
  res.include(center - temp);

  temp[0] = 0.f;
  temp[1] = radius;
  temp[2] = 0.f;

  res.include(center + temp);
  res.include(center - temp);

  temp[0] = 0.f;
  temp[1] = 0.f;
  temp[2] = radius;

  res.include(center + temp);
  res.include(center - temp);

  return res;
}
// I need to consolidate all this
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

}// namespace detail

Billboard::Billboard(const std::vector<std::string> &texts,
                     const std::vector<Vec<float32,3>> &positions)
  : m_up({0.f, 1.f, 0.f})
{
  Font *font = FontFactory::font("OpenSans-Regular");
  ScreenTextAnnotator anot;

  Array<float32> texture;
  int32 twidth,theight;
  Array<AABB<2>> tboxs, pboxs;
  anot.render_to_texture(texts, texture, twidth, theight, tboxs, pboxs);
  // world size of the font
  const float32 world_size = 10.f;
  const int32 size = texts.size();

  Array<Vec<float32,3>> centers;
  Array<Vec<float32,2>> dims;
  Array<Vec<float32,2>> tcoords;
  Array<AABB<3>> aabbs;

  centers.resize(size);
  dims.resize(size);
  tcoords.resize(size*4); // this is a quad
  aabbs.resize(size);

  Vec<float32,3> *centers_ptr = centers.get_host_ptr();
  Vec<float32,2> *dims_ptr = dims.get_host_ptr();
  Vec<float32,2> *tcoords_ptr = tcoords.get_host_ptr();
  AABB<3> *aabbs_ptr = aabbs.get_host_ptr();
  const AABB<2> *pbox_ptr = pboxs.get_host_ptr_const();
  const AABB<2> *tbox_ptr = tboxs.get_host_ptr_const();

  for(int i = 0; i < size; ++i)
  {
    Vec<float32,3> world_center = positions[i];
    Vec<float32,2> width_height;
    width_height[0] = pbox_ptr[i].m_ranges[0].length() * world_size;
    width_height[1] = pbox_ptr[i].m_ranges[1].length() * world_size;
    // To construct the bounding box for the BVH, just take the max dim and
    // use that as the radius of a sphere
    float32 radius = std::max(width_height[0], width_height[1]) / 2.f;

    aabbs_ptr[i] = detail::bound_sphere(world_center, radius);
    centers_ptr[i] = world_center;
    dims_ptr[i] = width_height;

    AABB<2> t_box = tbox_ptr[i];

    Vec<float32,2> bottom_left;
    bottom_left[0] = t_box.m_ranges[0].min();
    bottom_left[1] = t_box.m_ranges[1].min();

    Vec<float32,2> bottom_right;
    bottom_right[0] = t_box.m_ranges[0].max();
    bottom_right[1] = t_box.m_ranges[1].min();

    Vec<float32,2> top_left;
    top_left[0] = t_box.m_ranges[0].min();
    top_left[1] = t_box.m_ranges[1].max();

    Vec<float32,2> top_right;
    top_right[0] = t_box.m_ranges[0].max();
    top_right[1] = t_box.m_ranges[1].max();

    const int32 toffset = i * 4;
    tcoords_ptr[toffset + 0] = bottom_left;
    tcoords_ptr[toffset + 1] = bottom_right;
    tcoords_ptr[toffset + 2] = top_left;
    tcoords_ptr[toffset + 3] = top_right;
  }

  LinearBVHBuilder builder;
  m_bvh = builder.construct (aabbs);
  m_centers = centers;
  m_dims = dims;
  m_tcoords = tcoords;
  m_texture = texture;
  m_texture_width = twidth;
  m_texture_height = theight;
}

DRAY_EXEC
float32 tblerp(const float32 s,
               const float32 t,
               const float32 *texture,
               const int32 texture_width,
               const int32 texture_height)
{
  // we now need to blerp
  Vec<int32,2> st_min, st_max;
  st_min[0] = clamp(int32(s), 0, texture_width - 1);
  st_min[1] = clamp(int32(t), 0, texture_height - 1);
  st_max[0] = clamp(st_min[0]+1, 0, texture_width - 1);
  st_max[1] = clamp(st_min[1]+1, 0, texture_height - 1);

  Vec<float32,4> vals;
  vals[0] = texture[st_min[1] * texture_width + st_min[0]];
  vals[1] = texture[st_min[1] * texture_width + st_max[0]];
  vals[2] = texture[st_max[1] * texture_width + st_min[0]];
  vals[3] = texture[st_max[1] * texture_width + st_max[0]];

  float32 dx = s - float32(st_min[0]);
  float32 dy = t - float32(st_min[1]);

  float32 x0 = lerp(vals[0], vals[1], dx);
  float32 x1 = lerp(vals[2], vals[3], dx);
  // this the signed distance to the glyph
  return lerp(x0, x1, dy);
}

void Billboard::shade(const Array<Ray> &rays, const Array<RayHit> &hits, Framebuffer &fb)
{
  DeviceFramebuffer d_framebuffer(fb);
  const float32 *texture_ptr = m_texture.get_device_ptr();

  const int32 twidth = m_texture_width;
  const int32 theight = m_texture_height;
  Vec<float32,2> *tx_coords_ptr = m_tcoords.get_device_ptr();

  const RayHit *hit_ptr = hits.get_device_ptr_const();
  const Ray *rays_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, hits.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    if (hit.m_hit_idx != -1)
    {
      const int32 pixel_id = rays_ptr[ii].m_pixel_id;
      const int32 index = hit.m_hit_idx;
      const int32 offset = index * 4;
      // TODO: If we specialize this for text, then we can turn this
      // into an AABB<2>
      Vec<float32,2> t0 = tx_coords_ptr[offset + 0]; // bl
      Vec<float32,2> t1 = tx_coords_ptr[offset + 1]; // br
      Vec<float32,2> t2 = tx_coords_ptr[offset + 2]; // tl
      Vec<float32,2> t3 = tx_coords_ptr[offset + 3]; // tr

      Vec<float32,2> uv = {{hit.m_ref_pt[0], hit.m_ref_pt[1]}};
      Vec<float32,2> top = lerp(t2,t3, uv[0]);
      Vec<float32,2> bot = lerp(t0,t1, uv[0]);
      // the texture is upside down
      Vec<float32,2> st = lerp(bot, top, uv[1]);
      st[0] *= twidth;
      st[1] *= theight;

      float32 distance = tblerp(st[0], st[1], texture_ptr, twidth, theight);

      // TODO: add ray differentials for texture filtering
      const float32 smoothing = 1.0f / 16.0f;
      float32 alpha = smoothstep(0.5f - smoothing, 0.5f + smoothing, distance);

      Vec<float32,4> fcolor = {{ 1.f, 1.f, 0.f, alpha }};

      //fcolor = {{clamp(uv[0],0.f,1.f),
      //           clamp(uv[1],0.f,1.f),
      //           0.f,
      //           1.f}};


      if(alpha > 0.01f)
      {
        d_framebuffer.m_colors[pixel_id] = fcolor;
        d_framebuffer.m_depths[pixel_id] = hit.m_dist;
      }
    }
  });
  DRAY_ERROR_CHECK();
}

AABB<3> Billboard::bounds()
{
  return m_bvh.m_bounds;
}

Array<RayHit> Billboard::intersect (const Array<Ray> &rays)
{
  const Vec<float32,3> *centers_ptr = m_centers.get_device_ptr();
  const Vec<float32,2> *dims_ptr = m_dims.get_device_ptr();

  const int32 *leaf_ptr = m_bvh.m_leaf_nodes.get_device_ptr_const ();
  const Vec<float32, 4> *inner_ptr = m_bvh.m_inner_nodes.get_device_ptr_const ();

  const Ray *ray_ptr = rays.get_device_ptr_const ();

  const int32 size = rays.size ();

  Array<RayHit> hits;
  hits.resize (size);
  const Vec<float32,3> up_dir = m_up;

  RayHit *hit_ptr = hits.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i)
  {

    Ray ray = ray_ptr[i];

    RayHit hit;
    hit.init();

    Float closest_dist = ray.m_far;
    Float min_dist = ray.m_near;
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
        bool right_closer = detail::intersect_AABB (inner_ptr,
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

        const int32 bill_index = leaf_ptr[current_node];
        Vec<float32,3> center = centers_ptr[bill_index];
        Vec<float32,2> dims = dims_ptr[bill_index];


        //Ray tracing Gems II billboard intersections
        Vec<float32,3> normal_dir = ray.m_orig - center;
        bool y_align = false;
        Vec<float32,3> n = normal_dir;
        if(y_align)
        {
          n = {{normal_dir[0], 0.f, normal_dir[2]}};
        }

        n.normalize();
        Vec<float32,3> t = cross(up_dir, n);
        t.normalize();
        Vec<float32,3> b = cross(n, t);

        Matrix<float32,3,3> to_tangent;
        to_tangent.set_col(0,t);
        to_tangent.set_col(1,b);
        to_tangent.set_col(2,n);
        to_tangent = to_tangent.transpose();

        Vec<float32,3> dp = to_tangent * ray.m_dir;
        Vec<float32,3> op = to_tangent * normal_dir;
        float32 s = -op[2] / dp[2];
        Vec<float32,2> pp; // point on billboard plane
        pp[0] = op[0] + s * dp[0];
        pp[1] = op[1] + s * dp[1];

        //std::cout<<"distance "<<s<<"\n";
        bool hit_plane = s < closest_dist && s > min_dist;
        // check if the plane coordinates are within the dims
        if(hit_plane)
        {
          hit_plane = abs(pp[0]) < 0.5f * dims[0];
        }
        if(hit_plane)
        {
          hit_plane = abs(pp[1]) < 0.5f * dims[1];
        }
        if(hit_plane)
        {
          hit.m_ref_pt[0] = pp[0] / dims[0] + 0.5f;
          hit.m_ref_pt[1] = pp[1] / dims[1] + 0.5f;
          hit.m_dist = s;
          hit.m_hit_idx = bill_index;
          closest_dist = s;
        }

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } // while

    hit_ptr[i] = hit;
  });
  DRAY_ERROR_CHECK();
  return hits;
}

void Billboard::up(const Vec<float32,3> &up_dir)
{
  m_up = up_dir;
}

Vec<float32,3> Billboard::up() const
{
  return m_up;
}

}; //namepspace dray

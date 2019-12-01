#include <dray/filters/isosurface.hpp>
#include <dray/filters/internal/get_fragments.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/array_utils.hpp>
#include <dray/error.hpp>
#include <dray/device_framebuffer.hpp>
#include <dray/fragment.hpp>
#include <dray/isosurface_intersection.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/data_logger.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

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

void init_hits(Array<RayHit> &hits)
{
  const int32 size = hits.size();
  RayHit * hits_ptr = hits.get_device_ptr();
  Vec<Float,3> the_ninety_nine = {-99, 99, -99};

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
  {
    RayHit hit;
    hit.m_hit_idx = -1;
    hit.m_ref_pt = the_ninety_nine;
    hits_ptr[i] = hit;
  });
}

// Copied from point_location.hpp.
struct Candidates
{
  Array<int32> m_candidates;
  Array<int32> m_aabb_ids;
};

//
// candidate_ray_intersection()
//
//
template <class ElemT, int32 max_candidates>
Candidates candidate_ray_intersection(Array<Ray> rays,
                                      const BVH bvh,
                                      Field<FieldOn<ElemT, 1u>> &field,
                                      const float32 &iso_val)
{
  const int32 size = rays.size();

  Array<int32> candidates;
  Array<int32> aabb_ids;
  candidates.resize(size * max_candidates);
  aabb_ids.resize(size * max_candidates);
  array_memset(candidates, -1);

  //const int32 *active_ray_ptr = rays.m_active_rays.get_device_ptr_const();
  const Ray *ray_ptr = rays.get_device_ptr_const();

  const int32 *leaf_ptr = bvh.m_leaf_nodes.get_device_ptr_const();
  const Vec<float32, 4> *inner_ptr = bvh.m_inner_nodes.get_device_ptr_const();
  const int32 *aabb_ids_ptr = bvh.m_aabb_ids.get_device_ptr_const();

  DeviceField<FieldOn<ElemT, 1u>> device_field(field);
  int32 *candidates_ptr = candidates.get_device_ptr();
  int32 *cand_aabb_id_ptr = aabb_ids.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray &ray = ray_ptr[i];

    Float closest_dist = ray.m_far;
    Float min_dist = ray.m_near;
    ///int32 hit_idx = -1;
    const Vec<Float,3> dir = ray.m_dir;
    Vec<Float,3> inv_dir;
    inv_dir[0] = rcp_safe(dir[0]);
    inv_dir[1] = rcp_safe(dir[1]);
    inv_dir[2] = rcp_safe(dir[2]);

    int32 current_node;
    int32 todo[max_candidates];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;

    const Vec<Float,3> orig = ray.m_orig;

    Vec<Float,3> orig_dir;
    orig_dir[0] = orig[0] * inv_dir[0];
    orig_dir[1] = orig[1] * inv_dir[1];
    orig_dir[2] = orig[2] * inv_dir[2];

    int32 candidate_idx = 0;

    while (current_node != barrier && candidate_idx < max_candidates)
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
        AABB<1u> aabb_range;
        device_field.get_elem(el_idx).get_bounds(aabb_range);
        Range<> range = aabb_range.m_ranges[0];
        if(iso_val >= range.min() && iso_val <= range.max())
        {
          // Any leaf bbox we enter is a candidate.
          candidates_ptr[candidate_idx + i * max_candidates] = el_idx;
          cand_aabb_id_ptr[candidate_idx + i * max_candidates] = aabb_ids_ptr[current_node];
          candidate_idx++;
        }

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } //while

  });

  Candidates res;
  res.m_candidates = candidates;
  res.m_aabb_ids = aabb_ids;
  return res;
}

template <class ElemT>
void
intersect_isosurface(Array<Ray> rays,
                     float32 isoval,
                     Field<FieldOn<ElemT, 1u>> &field,
                     Mesh<ElemT> &mesh,
                     Array<RayHit> &hits)
{
  // This method intersects rays with the isosurface using the Newton-Raphson method.
  // The system of equations to be solved is composed from
  //   ** Transformations **
  //   1. PHI(u,v,w)  -- mesh element transformation, from ref space to R3.
  //   2. F(u,v,w)    -- scalar field element transformation, from ref space to R1.
  //   3. r(s)        -- ray parameterized by distance, relative to ray origin.
  //                     (We only restrict s >= 0. No expectation of s <= 1.)
  //   ** Targets **
  //   4. F_0         -- isovalue.
  //   5. Orig        -- ray origin.
  //
  // The ray-isosurface intersection is a solution to the following system:
  //
  // [ [PHI(u,v,w)]   [r(s)]         [ [      ]
  //   [          ] - [    ]     ==    [ Orig ]
  //   [          ]   [    ]           [      ]
  //   F(u,v,w)     +   0    ]           F_0    ]

  // Initialize outputs.
  //init_hits(hits); // TODO: not clear why we can't init inside main function

  const Vec<Float,3> element_guess = {0.5, 0.5, 0.5};
  const Float ray_guess = 1.0;

  // 1. Get intersection candidates for all active rays.
  constexpr int32 max_candidates = 64;
  Candidates candidates =
    candidate_ray_intersection<ElemT, max_candidates> (rays, mesh.get_bvh(), field, isoval);

  const int32    *cell_id_ptr = candidates.m_candidates.get_device_ptr_const();
  const int32    *aabb_id_ptr = candidates.m_aabb_ids.get_device_ptr_const();
  const AABB<3> *ref_aabb_ptr = mesh.get_ref_aabbs().get_device_ptr_const();

  const int32 size = rays.size();

    // Define pointers for RAJA kernel.
  DeviceMesh<ElemT> device_mesh(mesh);

  DeviceField<FieldOn<ElemT, 1u>> device_field(field);
  Ray *ray_ptr = rays.get_device_ptr();
  RayHit *hit_ptr = hits.get_device_ptr();

  Array<stats::Stats> mstats;
  mstats.resize(size);
  stats::Stats *mstats_ptr = mstats.get_device_ptr();

  // 4. For each active ray, loop through candidates until found an isosurface intersection.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
  {
    // TODO: don't make these references to main mem
    const Ray &ray = ray_ptr[i];
    RayHit hit;
    hit.m_hit_idx = -1;
    hit.m_dist = infinity<Float>();

    stats::Stats mstat;
    mstat.construct();

    Vec<Float,3> ref_coords = element_guess;
    Float ray_dist = ray_guess;

    bool found_inside = false;
    int32 candidate_idx = 0;
    int32 el_idx = cell_id_ptr[i*max_candidates + candidate_idx];
    int32 aabb_idx = aabb_id_ptr[i*max_candidates + candidate_idx];
    AABB<3> ref_start_box = ref_aabb_ptr[aabb_idx];
    int32 steps_taken = 0;
    while (!found_inside && candidate_idx < max_candidates && el_idx != -1)
    {
      ref_coords = element_guess;
      ray_dist = ray_guess;
      const bool use_init_guess = true;

      mstat.acc_candidates(1);

      found_inside =
        Intersector_RayIsosurf<ElemT>::intersect(mstat,
                                                 device_mesh.get_elem(el_idx),
                                                 device_field.get_elem(el_idx),
                                                 ray,
                                                 isoval,
                                                 ref_start_box,
                                                 ref_coords,
                                                 ray_dist,
                                                 use_init_guess);

      //steps_taken = iter_prof.m_num_iter;
      mstat.acc_iters(steps_taken);

      //TODO intersect multiple candidates and pick the nearest one.

      if (found_inside)
        break;
      else
      {
        // Continue searching with the next candidate.
        candidate_idx++;
        el_idx = cell_id_ptr[i*max_candidates + candidate_idx];
        aabb_idx = aabb_id_ptr[i*max_candidates + candidate_idx];
        ref_start_box = ref_aabb_ptr[aabb_idx];
      }

    } // end while



    if (found_inside)
    {
      hit.m_hit_idx = el_idx;
      hit.m_ref_pt = ref_coords;
      hit.m_dist = ray_dist;
      mstat.found();
    }

    mstats_ptr[i] = mstat;
    hit_ptr[i] = hit;
  });  // end RAJA

  stats::StatStore::add_ray_stats(rays, mstats);
}

}

Isosurface::Isosurface()
  : m_color_table("ColdAndHot"),
    m_iso_value(infinity32())
{
}

template<class ElemT>
void
Isosurface::execute(DataSet<ElemT> &data_set,
                    Array<Ray> &rays,
                    Framebuffer &framebuffer)
{
  DRAY_LOG_OPEN("isosuface");
  //Array<Vec<float32, 4>> color_buffer;
  //color_buffer.resize(rays.size());
  //Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  //array_memset_vec(color_buffer, init_color);
  DeviceFramebuffer d_framebuffer(framebuffer);

  Mesh<ElemT> mesh = data_set.get_mesh();

  assert(m_field_name != "");
  Field<FieldOn<ElemT, 1u>> field = data_set.get_field(m_field_name);

  if(m_iso_value == infinity32())
  {
    throw DRayError("Isosurface: no iso value set");
  }

  const int32 num_elems = mesh.get_num_elem();

  Array<RayHit> hits;
  hits.resize(rays.size());

  //const RefPoint<3> invalid_refpt{ -1, {-1,-1,-1} };
  //array_memset(rpoints, invalid_refpt);

  // Intersect rays with isosurface.
  detail::intersect_isosurface(rays,
                               m_iso_value,
                               field,
                               mesh,
                               hits);

  Array<Fragment> fragments=
    internal::get_fragments(rays, field.get_range(), field, mesh, hits);

  ColorMap color_map;
  color_map.color_table(m_color_table);
  color_map.scalar_range(field.get_range());

  Shader::blend_surf(framebuffer, color_map, rays, hits, fragments);
  DRAY_LOG_CLOSE();
}

void
Isosurface::set_field(const std::string field_name)
{
 m_field_name = field_name;
}

void
Isosurface::set_color_table(const ColorTable &color_table)
{
  m_color_table = color_table;
}

void
Isosurface::set_iso_value(const float32 iso_value)
{
  m_iso_value = iso_value;
}

template
void
Isosurface::execute(DataSet<MeshElem<3u, ElemType::Quad, Order::General>> &data_set,
                    Array<Ray> &rays,
                    Framebuffer &framebuffer);

}//namespace dray


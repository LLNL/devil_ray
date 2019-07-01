#include <dray/filters/isosurface.hpp>
#include <dray/array_utils.hpp>
#include <dray/error.hpp>
#include <dray/high_order_intersection.hpp>
#include <dray/shaders.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

template <typename T>
Array<ShadingContext<T>>
get_shading_context(Array<Ray<T>> &rays,
                    Field<T> &field,
                    Mesh<T> &mesh,
                    Array<RefPoint<T,3>> &rpoints)
{
  // Ray (read)    RefPoint (read)      ShadingContext (write)
  // ---           -----             --------------
  // m_pixel_id    m_el_id           m_pixel_id
  // m_dir         m_el_coords       m_ray_dir
  // m_orig
  // m_dist                          m_hit_pt
  //                                 m_is_valid
  //                                 m_sample_val
  //                                 m_normal
  //                                 m_gradient_mag

  //TODO store gradient and jacobian into separate fields??
  //Want to be able to use same get_shading_context() whether
  //the task is to shade isosurface or boundary surface...
  //
  //No, it's probably better to make a different get_shading_context
  //that knows it is using the cross product of two derivatives as the surface normal.
  //One reason is that rays hit_idx should store an index into which
  //the face id is embedded, i.e. (face_id + 6*el_id).
  //
  //In that case, "normal" really means the normal we will use for shading.
  //So it is appropriate to flip the normal to align with view, in this function.
  //(If it didn't mean "normal", but rather "gradient," then we shouldn't flip.)
  const int32 size_rays = rays.size();
  //const int32 size_active_rays = rays.m_active_rays.size();

  Array<ShadingContext<T>> shading_ctx;
  shading_ctx.resize(size_rays);
  ShadingContext<T> *ctx_ptr = shading_ctx.get_device_ptr();

  // Adopt the fields (m_pixel_id) and (m_dir) from rays to intersection_ctx.
  //shading_ctx.m_pixel_id = rays.m_pixel_id;
  //shading_ctx.m_ray_dir = rays.m_dir;

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<T,3> one_two_three = {123., 123., 123.};

  const int32 size = rays.size();

  const Range<> field_range = field.get_range();
  const T field_min = field_range.min();
  const T field_range_rcp = rcp_safe( field_range.length() );

  const Ray<T> *ray_ptr = rays.get_device_ptr_const();
  const RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr_const();

  MeshAccess<T> device_mesh = mesh.access_device_mesh();
  FieldAccess<T> device_field = field.access_device_field();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    ShadingContext<T> ctx;
    // TODO: create struct initializers
    ctx.m_hit_pt = one_two_three;
    ctx.m_normal = one_two_three;
    ctx.m_sample_val = 3.14f;
    ctx.m_gradient_mag = 55.55f;

    const Ray<T> &ray = ray_ptr[i];
    const RefPoint<T,3> &rpt = rpoints_ptr[i];

    ctx.m_pixel_id = ray.m_pixel_id;
    ctx.m_ray_dir  = ray.m_dir;

    if (rpt.m_el_id == -1)
    {
      // There is no intersection.
      ctx.m_is_valid = 0;
    }
    else
    {
      // There is an intersection.
      ctx.m_is_valid = 1;
    }

    if(ctx.m_is_valid)
    {
      // Compute hit point using ray origin, direction, and distance.
      ctx.m_hit_pt = ray.m_orig + ray.m_dir * ray.m_dist;

      // Evaluate element transformation to get scalar field value and gradient.

      const int32 el_id = rpt.m_el_id;
      const Vec<T,3> ref_pt = rpt.m_el_coords;

      Vec<T,3> space_val;
      Vec<Vec<T,3>,3> space_deriv;
      device_mesh.get_elem(el_id).eval(ref_pt, space_val, space_deriv);

      Vec<T,1> field_val;
      Vec<Vec<T,1>,3> field_deriv;
      device_field.get_elem(el_id).eval(ref_pt, field_val, field_deriv);

      // Move derivatives into matrix form.
      Matrix<T,3,3> jacobian;
      Matrix<T,1,3> gradient_h;
      for (int32 rdim = 0; rdim < 3; rdim++)
      {
        jacobian.set_col(rdim, space_deriv[rdim]);
        gradient_h.set_col(rdim, field_deriv[rdim]);
      }

      // Compute spatial field gradient as g = gh * J_inv.
      bool inv_valid;
      const Matrix<T,3,3> j_inv = matrix_inverse(jacobian, inv_valid);
      //TODO How to handle the case that inv_valid == false?
      const Matrix<T,1,3> gradient_mat = gradient_h * j_inv;
      Vec<T,3> gradient = gradient_mat.get_row(0);

      // Output.
      // TODO: deffer this calculation into the shader
      // output actual scalar value and normal
      ctx.m_sample_val = (field_val[0] - field_min) * field_range_rcp;
      ctx.m_gradient_mag = gradient.magnitude();
      gradient.normalize();   //TODO What if the gradient is (0,0,0)?

      if (dot(gradient, ray.m_dir) > 0.0f)
      {
        gradient = -gradient;   //Flip back toward camera.
      }
      ctx.m_normal = gradient;
    }

    ctx_ptr[i] = ctx;

  });

  return shading_ctx;
}

template <typename T>
DRAY_EXEC_ONLY
bool intersect_AABB(const Vec<float32,4> *bvh,
                    const int32 &currentNode,
                    const Vec<T,3> &orig_dir,
                    const Vec<T,3> &inv_dir,
                    const T& closest_dist,
                    bool &hit_left,
                    bool &hit_right,
                    const T &min_dist) //Find hit after this distance
{
  Vec<float32, 4> first4  = const_get_vec4f(&bvh[currentNode + 0]);
  Vec<float32, 4> second4 = const_get_vec4f(&bvh[currentNode + 1]);
  Vec<float32, 4> third4  = const_get_vec4f(&bvh[currentNode + 2]);
  T xmin0 = first4[0] * inv_dir[0] - orig_dir[0];
  T ymin0 = first4[1] * inv_dir[1] - orig_dir[1];
  T zmin0 = first4[2] * inv_dir[2] - orig_dir[2];
  T xmax0 = first4[3] * inv_dir[0] - orig_dir[0];
  T ymax0 = second4[0] * inv_dir[1] - orig_dir[1];
  T zmax0 = second4[1] * inv_dir[2] - orig_dir[2];
  T min0 = fmaxf(
    fmaxf(fmaxf(fminf(ymin0, ymax0), fminf(xmin0, xmax0)), fminf(zmin0, zmax0)),
    min_dist);
  T max0 = fminf(
    fminf(fminf(fmaxf(ymin0, ymax0), fmaxf(xmin0, xmax0)), fmaxf(zmin0, zmax0)),
    closest_dist);
  hit_left = (max0 >= min0);

  T xmin1 = second4[2] * inv_dir[0] - orig_dir[0];
  T ymin1 = second4[3] * inv_dir[1] - orig_dir[1];
  T zmin1 = third4[0] * inv_dir[2] - orig_dir[2];
  T xmax1 = third4[1] * inv_dir[0] - orig_dir[0];
  T ymax1 = third4[2] * inv_dir[1] - orig_dir[1];
  T zmax1 = third4[3] * inv_dir[2] - orig_dir[2];

  T min1 = fmaxf(
    fmaxf(fmaxf(fminf(ymin1, ymax1), fminf(xmin1, xmax1)), fminf(zmin1, zmax1)),
    min_dist);
  T max1 = fminf(
    fminf(fminf(fmaxf(ymin1, ymax1), fmaxf(xmin1, xmax1)), fmaxf(zmin1, zmax1)),
    closest_dist);
  hit_right = (max1 >= min1);

  return (min0 > min1);
}

//
// candidate_ray_intersection()
//
//
template <typename T, int32 max_candidates>
Array<int32> candidate_ray_intersection(Array<Ray<T>> rays, const BVH bvh)
{
  const int32 size = rays.size();

  Array<int32> candidates;
  candidates.resize(size * max_candidates);
  array_memset(candidates, -1);

  //const int32 *active_ray_ptr = rays.m_active_rays.get_device_ptr_const();
  const Ray<T> *ray_ptr = rays.get_device_ptr_const();

  const int32 *leaf_ptr = bvh.m_leaf_nodes.get_device_ptr_const();
  const Vec<float32, 4> *inner_ptr = bvh.m_inner_nodes.get_device_ptr_const();

  int32 *candidates_ptr = candidates.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray<T> &ray = ray_ptr[i];

    T closest_dist = ray.m_far;
    T min_dist = ray.m_near;
    ///int32 hit_idx = -1;
    const Vec<T,3> dir = ray.m_dir;
    Vec<T,3> inv_dir;
    inv_dir[0] = rcp_safe(dir[0]);
    inv_dir[1] = rcp_safe(dir[1]);
    inv_dir[2] = rcp_safe(dir[2]);

    int32 current_node;
    int32 todo[max_candidates];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;

    const Vec<T,3> orig = ray.m_orig;

    Vec<T,3> orig_dir;
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

        // Any leaf bbox we enter is a candidate.
        candidates_ptr[candidate_idx + i * max_candidates] = leaf_ptr[current_node];
        candidate_idx++;

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } //while

  });

  return candidates;
}

template <typename T, class StatsType>
void
intersect_isosurface(Array<Ray<T>> rays,
                     float32 isoval,
                     Field<T> &field,
                     Mesh<T> &mesh,
                     Array<RefPoint<T,3>> &rpoints,
                     StatsType &stats)
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
  {
    Vec<T,3> the_ninety_nine = {-99, 99, -99};
    Ray<T> *ray_ptr = rays.get_device_ptr();
    RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr();
    const int32 ray_size = rays.size();
    // TODO: HitIdx should already be set
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size),
                            [=] DRAY_LAMBDA (const int32 i)
    {
      RefPoint<T, 3> &rpt = rpoints_ptr[i];
      rpt.m_el_id = -1;
      rpt.m_el_coords = the_ninety_nine;
    });
  }

  const Vec<T,3> element_guess = {0.5, 0.5, 0.5};
  const T ray_guess = 1.0;

  // 1. Get intersection candidates for all active rays.
  constexpr int32 max_candidates = 64;
  Array<int32> candidates =
    candidate_ray_intersection<T, max_candidates> (rays, mesh.get_bvh());
  const int32 *candidates_ptr = candidates.get_device_ptr_const();

  const int32 size = rays.size();

    // Define pointers for RAJA kernel.
  MeshAccess<T> device_mesh = mesh.access_device_mesh();
  FieldAccess<T> device_field = field.access_device_field();
  Ray<T> *ray_ptr = rays.get_device_ptr();
  RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr();

#ifdef DRAY_STATS
  stats::AppStatsAccess device_appstats = stats.get_device_appstats();
#endif

  // 4. For each active ray, loop through candidates until found an isosurface intersection.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
  {
    Ray<T> &ray = ray_ptr[i];
    RefPoint<T,3> &rpt = rpoints_ptr[i];

    Vec<T,3> ref_coords = element_guess;
    T ray_dist = ray_guess;

    bool found_inside = false;
    int32 candidate_idx = 0;
    int32 el_idx = candidates_ptr[i*max_candidates + candidate_idx];
    int32 steps_taken = 0;
    while (!found_inside && candidate_idx < max_candidates && el_idx != -1)
    {
      ref_coords = element_guess;
      ray_dist = ray_guess;
      const bool use_init_guess = true;

#ifdef DRAY_STATS
      stats::IterativeProfile iter_prof;    iter_prof.construct();

      found_inside =
        Intersector_RayIsosurf<T>::intersect(iter_prof,
                                             device_mesh,
                                             device_field,
                                             el_idx,
                                             ray, isoval,
                                             ref_coords,
                                             ray_dist,
                                             use_init_guess);

      steps_taken = iter_prof.m_num_iter;
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_tests, 1);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_test_iterations, steps_taken);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_tests, 1);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_test_iterations, steps_taken);
#else
      found_inside = Intersector_RayIsosurf<T>::intersect(device_mesh, device_field, el_idx,   // Much easier than before.
        ray, isoval,
        ref_coords, ray_dist, use_init_guess);
#endif

      if (found_inside)
        break;
      else
      {
        // Continue searching with the next candidate.
        candidate_idx++;
        el_idx = candidates_ptr[i*max_candidates + candidate_idx];
      }

    } // end while


    if (found_inside)
    {
      rpt.m_el_id = el_idx;
      rpt.m_el_coords = ref_coords;
      ray.m_dist = ray_dist;
    }
    else
    {
      ray.m_active = 0;
      ray.m_dist = infinity<T>();
    }

#ifdef DRAY_STATS
    if (found_inside)
    {
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_hits, 1);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_hit_iterations, steps_taken);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_hits, 1);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_hit_iterations, steps_taken);
    }
#endif
  });  // end RAJA
}

}

Isosurface::Isosurface()
  : m_color_table("ColdAndHot"),
    m_iso_value(infinity32())
{
  m_color_table.add_alpha(0.0000, .1f);
  m_color_table.add_alpha(1.0000, .2f);
}

template<typename T>
Array<Vec<float32,4>>
Isosurface::execute(Array<Ray<T>> &rays,
                          DataSet<T> &data_set)
{
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());
  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  array_memset_vec(color_buffer, init_color);

  Mesh<T,3> mesh = data_set.get_mesh();

  assert(m_field_name != "");
  Field<T> field = data_set.get_field(m_field_name);

  if(m_iso_value == infinity32())
  {
    throw DRayError("Isosurface: no iso value set");
  }

  const int32 num_elems = mesh.get_num_elem();
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

  // Intersect rays with isosurface.
#ifdef DRAY_STATS
  detail::intersect_isosurface(rays,
                               m_iso_value,
                               field,
                               mesh,
                               rpoints,
                               *app_stats_ptr);
#else
  detail::intersect_isosurface(rays, m_iso_value, rpoints);
#endif

  Array<ShadingContext<T>> shading_ctx =
    detail::get_shading_context(rays, field, mesh, rpoints);

  dray::Shader::set_color_table(m_color_table);
  Shader::blend_surf(color_buffer, shading_ctx);

  return color_buffer;
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
Array<Vec<float32,4>>
Isosurface::execute<float32>(Array<Ray<float32>> &rays,
                                   DataSet<float32> &data_set);

template
Array<Vec<float32,4>>
Isosurface::execute<float64>(Array<Ray<float64>> &rays,
                                   DataSet<float64> &data_set);

}//namespace dray


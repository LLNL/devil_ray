#include <dray/high_order_shape.hpp>
#include <dray/array_utils.hpp>
#include <dray/color_table.hpp>
#include <dray/exports.hpp>
#include <dray/high_order_intersection.hpp>
#include <dray/newton_solver.hpp>
#include <dray/policies.hpp>
#include <dray/point_location.hpp>
#include <dray/simple_tensor.hpp>
#include <dray/shaders.hpp>
#include <dray/ref_point.hpp>
#include <dray/types.hpp>

#include <dray/utils/stats.hpp>

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>

#include <assert.h>
#include <iostream>
#include <stdio.h>

namespace dray
{


IsoBVH::IsoBVH(BVH &bvh, Range<> filter_range)
{
  m_inner_nodes.resize( bvh.m_inner_nodes.size() );
  m_leaf_nodes.resize( bvh.m_leaf_nodes.size() );
  array_copy(m_inner_nodes, bvh.m_inner_nodes);
  array_copy(m_leaf_nodes, bvh.m_leaf_nodes);
  m_bounds = bvh.m_bounds;

  m_filter_range = filter_range;
}




// ----------------------------------
// Support for MeshField::integrate()
// ----------------------------------


///
// TODO  Make common source for these detail methods, which are shared by MeshField and MFEMVolumeIntegrator.
//
namespace detail
{

template<typename T>
struct IsNonnegative  // Unary functor
{
  DRAY_EXEC bool operator() (const T val) const { return val >= 0; }
};

// Return false if the next candidate is -1.
// Return false if the reference coordinate is in bounds.
// Return true if the reference coordinate is Outside/Unknown and there is a next candidate.
template<class ShapeT, typename RefT, typename CandT, typename SolveT>
struct IsSearchable  // Ternary functor
{
    // Hack: NotConverged == 0.
  DRAY_EXEC bool operator() (const RefT ref_pt, const CandT cand_idx, const SolveT stat) const
  {
    return (cand_idx < 0) ? false :
        //(stat != SolveT::NotConverged && ShapeT::IsInside(ref_pt)) ? false : true;
        (stat != 0 && ShapeT::IsInside(ref_pt)) ? false : true;
  }
};


// Binary functor IsLess
//
template <typename T>
struct IsLess
{
  DRAY_EXEC bool operator()(const T &dist, const T &far) const
  {
    return dist < far;
  }
};

} // namespace detail


//
// MeshField::construct_bvh()
//
template <typename T>
BVH MeshField<T>::construct_bvh()
{
  constexpr double bbox_scale = 1.000001;

  const int num_els = m_size_el;

  constexpr int splits = 1;

  Array<AABB<>> aabbs;
  Array<int32> prim_ids;

  aabbs.resize(num_els*(splits+1));
  prim_ids.resize(num_els*(splits+1));

  AABB<> *aabb_ptr = aabbs.get_device_ptr();
  int32  *prim_ids_ptr = prim_ids.get_device_ptr();

  MeshAccess<T> device_mesh = this->m_mesh.access_device_mesh();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id)
  {
    AABB<> boxs[splits + 1];
    AABB<> ref_boxs[splits + 1];
    AABB<> tot;


    device_mesh.get_elem(el_id).get_bounds(boxs[0].m_ranges);
    tot = boxs[0];
    ref_boxs[0] = AABB<>::ref_universe();
    int32 count = 1;

    for(int i = 0; i < splits; ++i)
    {
      //find split
      int32 max_id = 0;
      float32 max_length = boxs[0].max_length();
      for(int b = 1; b < count; ++b)
      {
        float32 length = boxs[b].max_length();
        if(length > max_length)
        {
          max_id = b;
          max_length = length;
        }
      }

      int32 max_dim = boxs[max_id].max_dim();
      // split the reference box into two peices along largest phys dim
      ref_boxs[count] = ref_boxs[max_id].split(max_dim);

      // udpate the phys bounds
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[max_id].m_ranges,
                                                 boxs[max_id].m_ranges);
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[count].m_ranges,
                                                 boxs[count].m_ranges);
      count++;
    }

    AABB<> res;
    for(int i = 0; i < splits+1; ++i)
    {
      boxs[i].scale(bbox_scale);
      res.include(boxs[i]);
      aabb_ptr[el_id * (splits + 1) + i] = boxs[i];
      prim_ids_ptr[el_id * (splits + 1) + i] = el_id;
    }
    if(el_id > 100 && el_id < 200)
    {
      printf("cell id %d AREA %f %f diff %f\n",
                                     el_id,
                                     tot.area(),
                                     res.area(),
                                     tot.area() - res.area());
      //AABB<> ol =  tot.intersect(res);
      //float32 overlap =  ol.area();

      //printf("overlap %f\n", overlap);
      //printf("%f %f %f - %f %f %f\n",
      //      tot.m_ranges[0].min(),
      //      tot.m_ranges[1].min(),
      //      tot.m_ranges[2].min(),
      //      tot.m_ranges[0].max(),
      //      tot.m_ranges[1].max(),
      //      tot.m_ranges[2].max());
    }

  });

  LinearBVHBuilder builder;
  BVH bvh = builder.construct(aabbs, prim_ids);
  std::cout<<"****** "<<bvh.m_bounds<<" "<<bvh.m_bounds.area()<<"\n";
  return bvh;
}

//
// MeshField::field_bounds()
//
template <typename T>
void MeshField<T>::field_bounds(Range<> &scalar_range) // TODO move this capability into the bvh structure.
{
  // The idea is...
  // First assume that we have a positive basis.
  // Then the global maximum and minimum are guaranteed to be found on nodes/vertices.

  RAJA::ReduceMin<reduce_policy, T> comp_min(infinity32());
  RAJA::ReduceMax<reduce_policy, T> comp_max(neg_infinity32());

  const int32 num_nodes = m_field.get_dof_data().m_values.size();
  const T *node_val_ptr = (const T*) m_field.get_dof_data().m_values.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_nodes), [=] DRAY_LAMBDA (int32 ii)
  {
    comp_min.min(node_val_ptr[ii]);
    comp_max.max(node_val_ptr[ii]);
  });

  scalar_range.include( comp_min.get() );
  scalar_range.include( comp_max.get() );
}

//TODO: get rig of Rays and use dedicated stats class



//
// MeshField::get_shading_context()
//
// Returns shading context of size rays.
// This keeps image-bound buffers aligned with rays.
// For inactive rays, the is_valid flag is set to false.
//
template <typename T>
Array<ShadingContext<T>>
MeshField<T>::get_shading_context(Array<Ray<T>> &rays, Array<RefPoint<T,ref_dim>> &rpoints) const
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

  using ShapeOpType = BShapeOp<ref_dim>;
  using SpaceTransType = ElTransOp<T, ShapeOpType, ElTransIter<T,space_dim> >;
  using FieldTransType = ElTransOp<T, ShapeOpType, ElTransIter<T,field_dim> >;

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

  const Range<> field_range = get_scalar_range();
  const T field_min = field_range.min();
  const T field_range_rcp = rcp_safe( field_range.length() );

  const Ray<T> *ray_ptr = rays.get_device_ptr_const();
  const RefPoint<T,ref_dim> *rpoints_ptr = rpoints.get_device_ptr_const();

  MeshAccess<T> device_mesh = this->m_mesh.access_device_mesh();   // This is how we should do just before RAJA loop.
  FieldAccess<T> device_field = this->m_field.access_device_field();   // This is how we should do just before RAJA loop.

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    ShadingContext<T> ctx;
    // TODO: create struct initializers
    ctx.m_hit_pt = one_two_three;
    ctx.m_normal = one_two_three;
    ctx.m_sample_val = 3.14f;
    ctx.m_gradient_mag = 55.55f;

    const Ray<T> &ray = ray_ptr[i];
    const RefPoint<T,ref_dim> &rpt = rpoints_ptr[i];

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
      for (int32 rdim = 0; rdim < ref_dim; rdim++)
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

namespace detail
{

//
// intersect_AABB()
//
// Copied verbatim from triangle_mesh.cpp
//
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
  //   TODO find appropriate place for this function. It is mostly copied from TriangleMesh
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

  struct HasCandidate
  {
    int32 m_max_candidates;
    const int32 *m_candidates_ptr;
    DRAY_EXEC bool operator() (int32 ii) const { return (m_candidates_ptr[ii * m_max_candidates] > -1); }
  };

}  // namespace detail

//
// MeshField::intersect_mesh_boundary()
//
template <typename T>
template <class StatsType>
void
MeshField<T>::intersect_mesh_boundary(Array<Ray<T>> rays, Array<RefPoint<T,ref_dim>> &rpoints, StatsType &stats)
{
  // Initialize outputs.
  {
    Vec<T,3> the_ninety_nine = {-99, 99, -99};
    Ray<T> *ray_ptr = rays.get_device_ptr();
    RefPoint<T,ref_dim> *rpoints_ptr = rpoints.get_device_ptr();
    const int32 ray_size = rays.size();
    // TODO: HitIdx should already be set
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size),
                            [=] DRAY_LAMBDA (const int32 i)
    {
      RefPoint<T, ref_dim> &rpt = rpoints_ptr[i];
      rpt.m_el_id = -1;
      rpt.m_el_coords = the_ninety_nine;
    });
  }

  const Vec<T,3> element_guess = {0.5, 0.5, 0.5};
  const T ray_guess = 1.0;

  // Get intersection candidates for all active rays.
  constexpr int32 max_candidates = 64;
  Array<int32> candidates = detail::candidate_ray_intersection<T, max_candidates> (rays, m_bvh);
  const int32 *candidates_ptr = candidates.get_device_ptr_const();

  const int32 size = rays.size();

    // Define pointers for RAJA kernel.
  MeshAccess<T> device_mesh = m_mesh.access_device_mesh();
  Ray<T> *ray_ptr = rays.get_device_ptr();
  RefPoint<T,ref_dim> *rpoints_ptr = rpoints.get_device_ptr();

#ifdef DRAY_STATS
  stats::AppStatsAccess device_appstats = stats.get_device_appstats();
#endif

  // For each active ray, loop through candidates until found an intersection.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
  {
    Ray<T> &ray = ray_ptr[i];
    RefPoint<T,ref_dim> &rpt = rpoints_ptr[i];

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

      MeshElem<T> mesh_elem = device_mesh.get_elem(el_idx);
      found_inside = Intersector_RayFace<T>::intersect(iter_prof, mesh_elem, ray,
          ref_coords, ray_dist, use_init_guess);

      steps_taken = iter_prof.m_num_iter;
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_tests, 1);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_query_stats_ptr[i].m_total_test_iterations, steps_taken);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_tests, 1);
      RAJA::atomic::atomicAdd<atomic_policy>(&device_appstats.m_elem_stats_ptr[el_idx].m_total_test_iterations, steps_taken);
#else

      //TODO after add this to the Intersector_RayFace interface.
      /// found_inside = Intersector_RayFace<T>::intersect(device_mesh.get_elem(el_idx), ray,
      ///     ref_coords, ray_dist, use_init_guess);
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

  fprintf(stderr, "Here!!\n");
}

// Make "static constexpr" work with some linkers.
template <typename T> constexpr int32 MeshField<T>::ref_dim;
template <typename T> constexpr int32 MeshField<T>::space_dim;
template <typename T> constexpr int32 MeshField<T>::field_dim;

// Explicit instantiations.
template class MeshField<float32>;
template class MeshField<float64>;

} // namespace dray

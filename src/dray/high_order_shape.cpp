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
#include <dray/types.hpp>

#include <dray/utils/stats.hpp>

#include <assert.h>
#include <iostream>
#include <stdio.h>

namespace dray
{


IsoBVH::IsoBVH(BVH &bvh, Range filter_range)
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
// MeshField::integrate()
//
template <typename T>
Array<Vec<float32,4>>
MeshField<T>::integrate(Array<Ray<T>> rays, T sample_dist) const
{
  calc_ray_start(rays, get_bounds());

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());
  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  Vec<float32,4> bg_color = make_vec4f(1.f,1.f,1.f,1.f);
  array_memset_vec(color_buffer, init_color);

  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  Array<int32> active_rays = active_indices(rays);

#ifdef DRAY_STATS
  Array<Ray<T>> stat_rays;
  stat_rays.resize(rays.size());
#endif

  int32 dbg_count_iter = 0;
  std::cout<<"active rays "<<active_rays.size()<<"\n";
  while(active_rays.size() > 0)
  {
    // Find elements and reference coordinates for the points.
    locate(active_rays, rays
#ifdef DRAY_STATS
    , stat_rays
#endif
    );

//    locate(rays.calc_tips(), rays.m_active_rays, rays.m_hit_idx, rays.m_hit_ref_pt
//#ifdef DRAY_STATS
//    , rays
//#endif
//    );

    // Retrieve shading information at those points (scalar field value, gradient).
    Array<ShadingContext<T>> shading_ctx = get_shading_context(rays);

    // shade and blend sample using shading context  with color buffer
    Shader::blend(color_buffer, shading_ctx);

    advance_ray(rays, sample_dist);

    active_rays = active_indices(rays);

    std::cout << "MeshField::integrate() - Finished iteration " << dbg_count_iter++ << std::endl;
  }
	Shader::composite_bg(color_buffer,bg_color);
  return color_buffer;
}

 ////
 ////    //
 ////    // MeshField::isosurface_gradient()
 ////    //
 ////    template <typename T>
 ////    Array<Vec<float32,4>>
 ////    MeshField<T>::isosurface_gradient(Ray<T> rays, T isoval) const
 ////    {
 ////      // set up a color table
 ////      ColorTable color_table("cool2warm");
 ////      color_table.add_alpha(0.f, 1.0f);   // Solid colors only, just have one layer, no compositing.
 ////      color_table.add_alpha(1.f, 1.0f);
 ////      Array<Vec<float32, 4>> color_map;
 ////      constexpr int color_samples = 1024;
 ////      color_table.sample(color_samples, color_map);
 ////
 ////      // Initialize the color buffer to (0,0,0,0).
 ////      Array<Vec<float32, 4>> color_buffer;
 ////      color_buffer.resize(rays.size());
 ////      Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
 ////      array_memset_vec(color_buffer, init_color);
 ////
 ////      // Initial compaction: Literally remove the rays which totally miss the mesh.
 ////      detail::calc_ray_start(rays, get_bounds());
 ////      rays.m_active_rays = compact(rays.m_active_rays, rays.m_dist, rays.m_far, detail::IsLess<T>());
 ////
 ////      // Intersect rays with isosurface.
 ////      intersect_isosurface(rays, isoval);
 ////
 ////      Array<int32> valid_rays = compact(rays.m_active_rays, rays.m_hit_idx, detail::IsNonnegative<int32>());
 ////      const int32 *valid_rays_ptr = valid_rays.get_device_ptr_const();
 ////
 ////      ShadingContext<T> shading_ctx = get_shading_context(rays);
 ////
 ////      // Get gradient magnitude relative to overall field.
 ////      Array<T> gradient_mag_rel;
 ////      gradient_mag_rel.resize(shading_ctx.size());
 ////      const T *gradient_mag_ptr = shading_ctx.m_gradient_mag.get_device_ptr_const();
 ////      T *gradient_mag_rel_ptr = gradient_mag_rel.get_device_ptr();
 ////      RAJA::ReduceMax<reduce_policy, T> grad_max(-1);
 ////
 ////        // Reduce phase.
 ////      RAJA::forall<for_policy>(RAJA::RangeSegment(0, valid_rays.size()), [=] DRAY_LAMBDA (int32 v_idx)
 ////      {
 ////        const int32 r_idx = valid_rays_ptr[v_idx];
 ////        grad_max.max(gradient_mag_ptr[r_idx]);
 ////      });
 ////      const T norm_fac = rcp_safe(grad_max.get());
 ////
 ////        // Multiply phase.
 ////      RAJA::forall<for_policy>(RAJA::RangeSegment(0, valid_rays.size()), [=] DRAY_LAMBDA (int32 v_idx)
 ////      {
 ////        const int32 r_idx = valid_rays_ptr[v_idx];
 ////        gradient_mag_rel_ptr[r_idx] = gradient_mag_ptr[r_idx] * norm_fac;
 ////      });
 ////
 ////      shading_ctx.m_sample_val = gradient_mag_rel;  // shade using the gradient magnitude intstead.
 ////
 ////      detail::blend(color_buffer, color_map, shading_ctx);
 ////
 ////      return color_buffer;
 ////    }


//
// MeshField::construct_bvh()
//
template <typename T>
BVH MeshField<T>::construct_bvh()
{
  constexpr double bbox_scale = 1.000001;

  const int num_els = m_size_el;
  const int32 el_dofs_space = m_eltrans_space.m_el_dofs;

  Array<AABB> aabbs;
  aabbs.resize(num_els);
  AABB *aabb_ptr = aabbs.get_device_ptr();

  const int32 *ctrl_idx_ptr_space = m_eltrans_space.m_ctrl_idx.get_device_ptr_const();
  const Vec<T,space_dim> *ctrl_val_ptr_space = m_eltrans_space.m_values.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 elem)
  {
    ElTransIter<T,space_dim> space_data_iter;
    space_data_iter.init_iter(ctrl_idx_ptr_space, ctrl_val_ptr_space, el_dofs_space, elem);

    // Add each dof of the element to the bbox
    // Note: positivity of Bernstein bases ensures that convex
    //       hull of element nodes contain entire element
    AABB bbox;
    ElTransData<T,space_dim>::get_elt_node_range(space_data_iter, el_dofs_space, (Range*) &bbox);

    // Slightly scale the bbox to account for numerical noise
    bbox.scale(bbox_scale);

    aabb_ptr[elem] = bbox;
  });

  LinearBVHBuilder builder;
  return builder.construct(aabbs);
}

//
// MeshField::field_bounds()
//
template <typename T>
void MeshField<T>::field_bounds(Range &scalar_range) const // TODO move this capability into the bvh structure.
{
  // The idea is...
  // First assume that we have a positive basis.
  // Then the global maximum and minimum are guaranteed to be found on nodes/vertices.

  RAJA::ReduceMin<reduce_policy, T> comp_min(infinity32());
  RAJA::ReduceMax<reduce_policy, T> comp_max(neg_infinity32());

  const int32 num_nodes = m_eltrans_field.m_values.size();
  const T *node_val_ptr = (const T*) m_eltrans_field.m_values.get_device_ptr_const();

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
// MeshField::locate()
//
//template<typename T>
//void
//MeshField<T>::locate(Array<int32> active_idx,
//                     Array<Ray<T> rays
//#ifdef DRAY_STATS
//    , Array<Ray<T>> &rays
//#endif
//   ) const
//{
//  const Array<int32> active_idx = array_counting(points.size(), 0,1);
//    // Assume that elt_ids and ref_pts are sized to same length as points.
//  locate(points, active_idx, elt_ids, ref_pts
//#ifdef DRAY_STATS
//    , rays
//#endif
//  );
//}

//
// MeshField::locate()
//
template<typename T>
void
MeshField<T>::locate(Array<int32> &active_idx,
                     Array<Ray<T>> &rays
#ifdef DRAY_STATS
    , Array<Ray<T>> &stat_rays
#endif
    ) const
{
  using ShapeOpType = BShapeOp<ref_dim>;
  using TransOpType = ElTransOp<T, ShapeOpType, ElTransIter<T,space_dim> >;

  const int32 size = rays.size();
  const int32 size_active = active_idx.size();
  const int32 size_aux = ShapeOpType::get_aux_req(m_p_space);
  ////const int32 el_dofs_space = m_eltrans_space.m_el_dofs;

  PointLocator locator(m_bvh);
  //constexpr int32 max_candidates = 5;
  constexpr int32 max_candidates = 100;
  Array<Vec<T,3>> points = calc_tips(rays);
  Array<int32> candidates = locator.locate_candidates(points, active_idx, max_candidates);  //Size size_active * max_candidates.

  Intersector_PointVol<T> _intersector = Intersector_PointVol<T>::factory(m_eltrans_space);

  // For now the initial guess will always be the center of the element. TODO
  Vec<T,ref_dim> _ref_center;
  _ref_center = 0.5f;
  const Vec<T,ref_dim> ref_center = _ref_center;

  // Initialize outputs to well-defined dummy values.
  constexpr Vec<T,3> three_point_one_four = {3.14, 3.14, 3.14};

  // Assume that elt_ids and ref_pts are sized to same length as points.
  //assert(elt_ids.size() == ref_pts.size());

  // Auxiliary memory for evaluating element transformations.
  Array<T> aux_array;
  aux_array.resize(size_aux * size_active);

  const int32    *active_idx_ptr = active_idx.get_device_ptr_const();

  Ray<T> *ray_ptr = rays.get_device_ptr();

  const Vec<T,3> *points_ptr     = points.get_device_ptr_const();
  const int32    *candidates_ptr = candidates.get_device_ptr_const();
  const int32    *ctrl_idx_ptr   = m_eltrans_space.m_ctrl_idx.get_device_ptr_const();
  const Vec<T,3> *ctrl_val_ptr   = m_eltrans_space.m_values.get_device_ptr_const();
  T        *aux_array_ptr = aux_array.get_device_ptr();

  const int32 p_space = m_p_space;
  ////constexpr typename NewtonSolve<T>::SolveStatus not_converged = NewtonSolve<T>::NotConverged; // local

#ifdef DRAY_STATS
  // Find out how many NewtonSolves we are taking per point.
  ///NewtonSolveHistogram stats = NewtonSolveHistogram::factory();
  Ray<T> *stat_ray_ptr = stat_rays.get_device_ptr();
#endif

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 ii = active_idx_ptr[aii];
    Ray<T> ray = ray_ptr[ii];

    ray.m_hit_ref_pt = three_point_one_four;
    ray.m_hit_idx = -1;

    const Vec<T,3> target_pt = ray.m_orig + ray.m_dir * ray.m_dist;
    // - Use aii to index into candidates.
    // - Use ii to index into points, elt_ids, and ref_pts.

    Intersector_PointVol<T> intersector = _intersector;   // Make a copy of the outer _intersector.

    int32 count = 0;
    int32 el_idx = candidates_ptr[aii*max_candidates + count];
    Vec<T,ref_dim> ref_pt = ref_center;

    T * const aux_mem_ptr = aux_array_ptr + aii * size_aux;
    TransOpType trans;
    trans.init_shape(p_space, aux_mem_ptr);

#ifdef DRAY_STATS
    NewtonSolveCounter newton_solve_counter;
#endif

    bool found_inside = false;
    while(!found_inside && count < max_candidates && el_idx != -1)
    {
      ////trans.m_coeff_iter.init_iter(ctrl_idx_ptr, ctrl_val_ptr, el_dofs_space, el_idx);
      ref_pt = ref_center;    // Initial guess.

      ////const float32 tol_phys = 0.00001;      // TODO
      ////const float32 tol_ref  = 0.00001;

      ////int32 steps_taken;
      ////typename NewtonSolve<T>::SolveStatus status = not_converged;
      ////status = NewtonSolve<T>::solve(trans, target_pt, ref_pt, tol_phys, tol_ref, steps_taken);

      ////if ( status != not_converged && ShapeOpType::is_inside(ref_pt) )

      T unused_dist;
      int32 steps_taken;
      intersector(el_idx, trans, target_pt, found_inside, steps_taken, unused_dist, ref_pt);

#ifdef DRAY_STATS
      newton_solve_counter.add_candidate_steps(steps_taken);
#endif

      if (found_inside)
      {
        // Found the element. Stop search, preserving count and el_idx.
        ////found_inside = true;
        break;
      }
      else
      {
        // Continue searching with the next candidate.
        count++;
        el_idx = candidates_ptr[aii*max_candidates + count];
           //NOTE: This may read past end of array, but only if count >= max_candidates.
      }
    }

#ifdef DRAY_STATS
    //newton_solve_counter.finalize_search(found_inside, stats);
    Ray<T> stat_ray;
    newton_solve_counter.finalize_search(found_inside,
                                         stat_ray.m_wasted_steps,
                                         stat_ray.m_total_steps);
    stat_ray_ptr[ii] = stat_ray;
#endif

    // After testing each candidate, now record the result.
    if (found_inside)
    {
      ray.m_hit_idx = el_idx;
      ray.m_hit_ref_pt = ref_pt;

    }
    else
    {
      ray.m_hit_idx = -1;
    }
    if(ray.m_dist >= ray.m_far) ray.m_active = 0;
    ray_ptr[ii] = ray;
  });
}

template <typename T>
DeviceFieldData<T>
MeshField<T>::get_device_field_data() const
{
  DeviceFieldData<T> res
  {
    m_eltrans_space.m_el_dofs,
    m_eltrans_field.m_el_dofs,
    m_eltrans_space.m_ctrl_idx.get_device_ptr_const(),
    m_eltrans_space.m_values.get_device_ptr_const(),
    m_eltrans_field.m_ctrl_idx.get_device_ptr_const(),
    m_eltrans_field.m_values.get_device_ptr_const(),
    m_p_space,
    m_p_field
  };
  return res;
}
//
// MeshField::get_shading_context()
//
// Returns shading context of size rays.
// This keeps image-bound buffers aligned with rays.
// For inactive rays, the is_valid flag is set to false.
//
template <typename T>
Array<ShadingContext<T>>
MeshField<T>::get_shading_context(Array<Ray<T>> &rays) const
{
  // Ray (read)                ShadingContext (write)
  // ---                       --------------
  // m_pixel_id                m_pixel_id
  // m_dir                     m_ray_dir
  // m_orig
  // m_dist                    m_hit_pt
  //                           m_is_valid
  // m_hit_idx                 m_sample_val
  // m_hit_ref_pt              m_normal
  //                           m_gradient_mag

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

  const Range field_range = get_scalar_range();
  const T field_min = field_range.min();
  const T field_range_rcp = rcp_safe( field_range.length() );

  const int32 size_aux_space = SpaceTransType::get_aux_req(m_p_space);
  const int32 size_aux_field = FieldTransType::get_aux_req(m_p_field);
  const int32 size_aux = max(size_aux_space, size_aux_field);
  // Auxiliary memory to help evaluate element transformations.
  Array<T> aux_array;
  aux_array.resize(size_aux * size);
  array_memset(aux_array, (T) -1.);   // Dummy value.

  const Ray<T> *ray_ptr = rays.get_device_ptr_const();

  const DeviceFieldData<T> field = get_device_field_data();
  T *aux_array_ptr = aux_array.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    ShadingContext<T> ctx;
    // TODO: create struct initializers
    ctx.m_hit_pt = one_two_three;
    ctx.m_normal = one_two_three;
    ctx.m_sample_val = 3.14f;
    ctx.m_gradient_mag = 55.55f;

    const Ray<T> &ray = ray_ptr[i];

    ctx.m_pixel_id = ray.m_pixel_id;
    ctx.m_ray_dir  = ray.m_dir;

    if (ray.m_hit_idx == -1)
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

      const int32 el_id = ray.m_hit_idx;
      const Vec<T,3> ref_pt = ray.m_hit_ref_pt;
      T * const aux_mem_ptr = aux_array_ptr + i * size_aux;   // size_aux is big enough for either transformation.

      Vec<T,3> space_val;
      Vec<Vec<T,3>,3> space_deriv;
      {
        SpaceTransType trans_space;
        trans_space.init_shape(field.m_p_space, aux_mem_ptr);
        // TODO: this should just take field or element can be initted
        // from field.get_element(el_id)
        trans_space.m_coeff_iter.init_iter(field.m_space_idx_ptr,
                                           field.m_space_val_ptr,
                                           field.m_el_dofs_space, el_id);
        trans_space.eval(ref_pt, space_val, space_deriv);
      }

      Vec<T,1> field_val;
      Vec<Vec<T,1>,3> field_deriv;
      {
        FieldTransType trans_field;
        trans_field.init_shape(field.m_p_field, aux_mem_ptr);
        // TODO: this should just take field or element can be initted
        // from field.get_element(el_id)
        trans_field.m_coeff_iter.init_iter(field.m_field_idx_ptr,
                                           field.m_field_val_ptr,
                                           field.m_el_dofs_field,
                                           el_id);
        trans_field.eval(ref_pt, field_val, field_deriv);
      }

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

// ---------------------------------------------
// Support for MeshField::intersect_isosurface()
// ---------------------------------------------

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

    ///T *dist_ptr = rays.m_dist.get_device_ptr();
    ///int32 *hit_idx_ptr = rays.m_hit_idx.get_device_ptr();

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

          /// // Set ray hit_idx and dist using the closest candidate.
          /// if (candidate_idx == 0)
          /// {
          ///   dist_ptr[i] = closest_dist;
          /// }

          current_node = todo[stackptr];
          stackptr--;
        } // if leaf node

      } //while

      /// // Set ray hit_idx using the closes (or nonexistent) candidate.
      /// hit_idx_ptr[i] = candidates_ptr[0 + aii*max_candidates];
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
// MeshField::intersect_isosurface()
//
template <typename T>
void
MeshField<T>::intersect_isosurface(Array<Ray<T>> rays, T isoval)
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
    const int32 ray_size = rays.size();
    // TODO: HitIdx should already be set
    //array_memset(rays.m_hit_idx, rays.m_active_rays, -1);
    //array_memset(rays.m_hit_ref_pt, rays.m_active_rays, the_ninety_nine);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size),
                            [=] DRAY_LAMBDA (const int32 i)
    {
      Ray<T> &ray = ray_ptr[i];
      ray.m_hit_idx = -1;
      ray.m_hit_ref_pt = the_ninety_nine;
    });
  }

  //Array<int32> active_rays = active_indices(rays);

  //std::cerr<<"start intersect_isosurface()\n";
  //std::cerr<<"rays.m_active_rays.size() == " << active_rays.size() << std::endl;

  using TransOp = ElTransRayOp<T, ElTransPairOp<T, SpaceTransOp, FieldTransOp>, space_dim>;
  const Vec<T,4> initial_guess = {0.5, 0.5, 0.5, 1.0};
  ///constexpr T tp = 0.00001, tf = 0.00001;   //TODO
  ///constexpr typename NewtonSolve<T>::SolveStatus not_converged = NewtonSolve<T>::NotConverged;

  // 1. Check if isoval is in range of cached m_iso_bvh; if not, construct & cache new m_iso_bvh.
  T iso_bvh_min = m_iso_bvh.m_filter_range.min();
  T iso_bvh_max = m_iso_bvh.m_filter_range.max();
  if (!(iso_bvh_min <= isoval && isoval <= iso_bvh_max))
  {
     constexpr T iso_margin = 0.05;  //TODO what should this be?
     Range iso_range;
     iso_range.include(isoval - iso_margin);
     iso_range.include(isoval + iso_margin);
     m_iso_bvh = construct_iso_bvh(iso_range);
  }

  // 2. Get intersection candidates for all active rays.
  constexpr int32 max_candidates = 64;
  Array<int32> candidates = detail::candidate_ray_intersection<T, max_candidates> (rays, m_iso_bvh);
  const int32 *candidates_ptr = candidates.get_device_ptr_const();

  //////  ///// 3. Filter active rays by those with at least one candidate.
  //////  ///detail::HasCandidate has_candidate;
  //////  ///has_candidate.m_max_candidates = max_candidates;
  //////  ///has_candidate.m_candidates_ptr = candidates_ptr;
  //////  ///Array<int32> arr_c = array_counting(rays.m_active_rays.size(), 0,1);
  //////  ///Array<int32> active_rays = gather(rays.m_active_rays, compact(arr_c, has_candidate));
  //////  ///printf("The number of rays with at least one candidate:  %d\n", compact(arr_c, has_candidate).size());
  //////  /////XXX The outcome is wrong because candidates must also be compacted.
  //////  ///const int32 size_active = active_rays.size();

  const int32 size = rays.size();

    // Sizes / Aux mem to evaluate transformations.
  ///const int32 el_dofs_space = m_eltrans_space.m_el_dofs;
  ///const int32 el_dofs_field = m_eltrans_field.m_el_dofs;
  const int32 size_aux = max(SpaceTransOp::get_aux_req(m_p_space),
                             FieldTransOp::get_aux_req(m_p_field));
  Array<T> aux_array;
  aux_array.resize(size_aux * size);

    // Local for lambda capture.
  constexpr int32 space_dim = MeshField<T>::space_dim;
  const int32 p_space = m_p_space;
  const int32 p_field = m_p_field;

  Intersector_RayIsosurf<T> _intersector = Intersector_RayIsosurf<T>::factory(m_eltrans_space, m_eltrans_field);

    // Define pointers for RAJA kernel.
  const int32            * const space_ctrl_ptr = m_eltrans_space.m_ctrl_idx.get_device_ptr_const();
  const Vec<T,space_dim> * const space_val_ptr = m_eltrans_space.m_values.get_device_ptr_const();
  const int32            * const field_ctrl_ptr = m_eltrans_field.m_ctrl_idx.get_device_ptr_const();
  const Vec<T,field_dim> * const field_val_ptr = m_eltrans_field.m_values.get_device_ptr_const();
  T * const aux_array_ptr = aux_array.get_device_ptr();

  Ray<T> *ray_ptr = rays.get_device_ptr();

#ifdef DRAY_STATS
  // Find out how many NewtonSolves we are taking per point.
  ///NewtonSolveHistogram stats = NewtonSolveHistogram::factory();
  //int32 *r_wasted_steps = rays.m_wasted_steps.get_device_ptr();
  //int32 *r_total_steps = rays.m_total_steps.get_device_ptr();
#endif

  // 4. For each active ray, loop through candidates until found an isosurface intersection.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
  {
    Intersector_RayIsosurf<T> intersector = _intersector;

    T * const aux_mem_ptr = aux_array_ptr + i * size_aux;
    Ray<T> &ray = ray_ptr[i];

    TransOp trans;
    trans.trans_x.init_shape(p_space, aux_mem_ptr);
    trans.trans_y.init_shape(p_field, aux_mem_ptr);
    trans.set_minus_ray_dir(ray.m_dir);

    Vec<T,4> ref_pt;

    ///Vec<T,4> _target;
    ///((Vec<T,3> &)_target).operator=(ray.m_orig);
    ///_target[3] = isoval;
    ///const Vec<T,4> &target = _target;

#ifdef DRAY_STATS
    NewtonSolveCounter newton_solve_counter;
#endif

    bool found_inside = false;
    int32 candidate_idx = 0;
    int32 el_idx = candidates_ptr[candidate_idx + i * max_candidates];
    while (!found_inside && candidate_idx < max_candidates && el_idx != -1)
    {
      /// //trans.trans_x.m_coeff_iter.init_iter(space_ctrl_ptr, space_val_ptr, el_dofs_space, el_idx);
      /// //trans.trans_y.m_coeff_iter.init_iter(field_ctrl_ptr, field_val_ptr, el_dofs_field, el_idx);

      ref_pt = initial_guess;

      /// //int32 steps;
      /// //typename NewtonSolve<T>::SolveStatus status = not_converged;
      /// //status = NewtonSolve<T>::solve(trans, target, ref_pt, tp,tf, steps);

      /// //if ( status != not_converged && BShapeOp<3>::is_inside((Vec<T,space_dim>&) ref_pt) && ref_pt[space_dim] > 0)
      T dist;
      int32 steps_taken;
      intersector(el_idx,
                  trans,
                  {ray.m_dir, ray.m_orig, isoval},
                  found_inside,
                  steps_taken,
                  dist,
                  ref_pt);

#ifdef DRAY_STATS
      newton_solve_counter.add_candidate_steps(steps_taken);
#endif

      if (found_inside)
      {
        ///found_inside = true;
        break;
      }
      else
      {
        candidate_idx++;
        el_idx = candidates_ptr[candidate_idx + i * max_candidates];
      }
    } // end while

#ifdef DRAY_STATS
    //newton_solve_counter.finalize_search(found_inside, stats);
    newton_solve_counter.finalize_search(found_inside,
                                         ray.m_wasted_steps,
                                         ray.m_total_steps);
#endif

    if (found_inside)
    {
      ray.m_hit_idx = el_idx;
      ray.m_hit_ref_pt = (Vec<T,space_dim> &) ref_pt;
      ray.m_dist = ref_pt[space_dim];
    }
    else
    {
      ray.m_active = 0;
    }
  });  // end RAJA
}


//
// MeshField::isosurface_gradient()
//
template <typename T>
Array<Vec<float32,4>>
MeshField<T>::isosurface_gradient(Array<Ray<T>> rays, T isoval)
{
  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());
  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  array_memset_vec(color_buffer, init_color);
  std::cerr<<"init\n";

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  calc_ray_start(rays, get_bounds());

  // Intersect rays with isosurface.
  intersect_isosurface(rays, isoval);

  Array<int32> active_rays = active_indices(rays);

  std::cerr<<"start intersect_isosurface()\n";
  std::cerr<<"rays.m_active_rays.size() == " << active_rays.size() << std::endl;

  std::cerr<<"compacted\n";
  Array<ShadingContext<T>> shading_ctx = get_shading_context(rays);

  // These are commented out so that we shade using the normalized scalar value.
  // It should produce a uniformly colored surface. But the color will be different
  // for different isovalues.
  //
  // ///  // Get gradient magnitude relative to overall field.
  // ///  Array<T> gradient_mag_rel;
  // ///  gradient_mag_rel.resize(shading_ctx.size());
  // ///  const T *gradient_mag_ptr = shading_ctx.m_gradient_mag.get_device_ptr_const();
  // ///  T *gradient_mag_rel_ptr = gradient_mag_rel.get_device_ptr();
  // ///  RAJA::ReduceMax<reduce_policy, T> grad_max(-1);
  // ///  std::cout<<"shading context\n";

  // ///    // Reduce phase.
  // ///  RAJA::forall<for_policy>(RAJA::RangeSegment(0, valid_rays.size()), [=] DRAY_LAMBDA (int32 v_idx)
  // ///  {
  // ///    const int32 r_idx = valid_rays_ptr[v_idx];
  // ///    grad_max.max(gradient_mag_ptr[r_idx]);
  // ///  });
  // ///  const T norm_fac = rcp_safe(grad_max.get());

  // ///    // Multiply phase.
  // ///  RAJA::forall<for_policy>(RAJA::RangeSegment(0, valid_rays.size()), [=] DRAY_LAMBDA (int32 v_idx)
  // ///  {
  // ///    const int32 r_idx = valid_rays_ptr[v_idx];
  // ///    gradient_mag_rel_ptr[r_idx] = gradient_mag_ptr[r_idx] * norm_fac;
  // ///  });

  // ///  shading_ctx.m_sample_val = gradient_mag_rel;  // shade using the gradient magnitude intstead.

  Shader::blend_surf(color_buffer, shading_ctx);

  return color_buffer;
}


// Make "static constexpr" work with some linkers.
template <typename T> constexpr int32 MeshField<T>::ref_dim;
template <typename T> constexpr int32 MeshField<T>::space_dim;
template <typename T> constexpr int32 MeshField<T>::field_dim;

// Explicit instantiations.
template class MeshField<float32>;
template class MeshField<float64>;




} // namespace dray

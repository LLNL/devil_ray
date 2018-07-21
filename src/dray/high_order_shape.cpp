#include <dray/high_order_shape.hpp>
#include <dray/array_utils.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>

#include <dray/color_table.hpp>
#include <dray/point_location.hpp>

#include <assert.h>
#include <iostream>
#include <stdio.h>

namespace dray
{

//
// ElTransData::resize()
//
template <typename T, int32 PhysDim>
void 
ElTransData<T,PhysDim>::resize(int32 size_el, int32 el_dofs, int32 size_ctrl)
{
  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_ctrl;

  m_ctrl_idx.resize(size_el * el_dofs);
  m_values.resize(size_ctrl);
}

template struct ElTransData<float32, 3>;
template struct ElTransData<float32, 1>;
template struct ElTransData<float64, 3>;
template struct ElTransData<float64, 1>;


// ----------------------------------
// Support for MeshField::integrate()
// ----------------------------------


///
// TODO  Make common source for these detail methods, which are shared by MeshField and MFEMVolumeIntegrator.
//
namespace detail
{
//
// utility function to find estimated intersection with the bounding
// box of the mesh
//
// After calling:
//   rays m_near   : set to estimated mesh entry
//   rays m_far    : set to estimated mesh exit
//   rays hit_idx  : -1 if ray missed the AABB and 1 if it hit
//
//   if ray missed then m_far <= m_near, if ray hit then m_far > m_near.
//
template<typename T>
void calc_ray_start(Ray<T> &rays, AABB bounds)
{
  // avoid lambda capture issues
  AABB mesh_bounds = bounds;

  const Vec<T,3> *dir_ptr = rays.m_dir.get_device_ptr_const();
  const Vec<T,3> *orig_ptr = rays.m_orig.get_device_ptr_const();

  T *near_ptr = rays.m_near.get_device_ptr();
  T *far_ptr  = rays.m_far.get_device_ptr();
  int32 *hit_idx_ptr = rays.m_hit_idx.get_device_ptr();

  const int32 size = rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<T,3> ray_dir = dir_ptr[i];
    Vec<T,3> ray_orig = orig_ptr[i];

    float32 dirx = static_cast<float32>(ray_dir[0]);
    float32 diry = static_cast<float32>(ray_dir[1]);
    float32 dirz = static_cast<float32>(ray_dir[2]);
    float32 origx = static_cast<float32>(ray_orig[0]);
    float32 origy = static_cast<float32>(ray_orig[1]);
    float32 origz = static_cast<float32>(ray_orig[2]);

    float32 inv_dirx = rcp_safe(dirx);
    float32 inv_diry = rcp_safe(diry);
    float32 inv_dirz = rcp_safe(dirz);

    float32 odirx = origx * inv_dirx;
    float32 odiry = origy * inv_diry;
    float32 odirz = origz * inv_dirz;

    float32 xmin = mesh_bounds.m_x.min() * inv_dirx - odirx;
    float32 ymin = mesh_bounds.m_y.min() * inv_diry - odiry;
    float32 zmin = mesh_bounds.m_z.min() * inv_dirz - odirz;
    float32 xmax = mesh_bounds.m_x.max() * inv_dirx - odirx;
    float32 ymax = mesh_bounds.m_y.max() * inv_diry - odiry;
    float32 zmax = mesh_bounds.m_z.max() * inv_dirz - odirz;

    const float32 min_int = 0.f;
    float32 min_dist = max(max(max(min(ymin, ymax), min(xmin, xmax)), min(zmin, zmax)), min_int);
    float32 max_dist = min(min(max(ymin, ymax), max(xmin, xmax)), max(zmin, zmax));

    int32 hit = -1; // miss flag

    if (max_dist > min_dist)
    {
      hit = 1; 
    }

    near_ptr[i] = min_dist;
    far_ptr[i] = max_dist;
    
    hit_idx_ptr[i] = hit;

  });
}

// this is a place holder function. Normally we could just set 
// values somewhere else, but for testing lets just do this now
template<typename T>
void advance_ray(Ray<T> &rays, float32 distance)
{
  // aviod lambda capture issues
  T dist = distance;

  /// const T *near_ptr = rays.m_near.get_device_ptr_const();
  /// const T *far_ptr  = rays.m_far.get_device_ptr_const();
  const int32 *active_ray_ptr = rays.m_active_rays.get_device_ptr_const();

  T *dist_ptr  = rays.m_dist.get_device_ptr();
  /// int32 *hit_idx_ptr = rays.m_hit_idx.get_device_ptr();

  const int32 size = rays.m_active_rays.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 ray_idx = active_ray_ptr[i];
    /// T far = far_ptr[ray_idx];
    T current_dist = dist_ptr[ray_idx];
    // advance ray
    current_dist += dist;
    dist_ptr[ray_idx] = current_dist;
     
  });
}


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

template<typename T>
void blend(Array<Vec4f> &color_buffer,
           Array<Vec4f> &color_map,
           ShadingContext<T> &shading_ctx)

{
  const int32 *pid_ptr = shading_ctx.m_pixel_id.get_device_ptr_const();
  const int32 *is_valid_ptr = shading_ctx.m_is_valid.get_device_ptr_const();
  const T *sample_val_ptr = shading_ctx.m_sample_val.get_device_ptr_const();

  const Vec<T,3> *normal_ptr = shading_ctx.m_normal.get_device_ptr_const();
  const Vec<T,3> *hit_pt_ptr = shading_ctx.m_hit_pt.get_device_ptr_const();
  const Vec<T,3> *ray_dir_ptr = shading_ctx.m_ray_dir.get_device_ptr_const();

  const Vec4f *color_map_ptr = color_map.get_device_ptr_const();

  Vec4f *img_ptr = color_buffer.get_device_ptr();

  const int color_map_size = color_map.size();


  Vec<float32,3> light_color = make_vec3f(1.f,1.f,1.f);
  Vec<float32,3> light_amb = make_vec3f(0.1f,0.1f,0.1f);
  Vec<float32,3> light_diff = make_vec3f(0.3f,0.3f,0.3f);
  Vec<float32,3> light_spec = make_vec3f(0.7f,0.7f,0.7f);
  float32 spec_pow = 80.0; //shiny

  Vec<T,3> light_pos;

  light_pos[0] = 20.f;
  light_pos[1] = 10.f;
  light_pos[2] = 50.f;

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, shading_ctx.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    if (is_valid_ptr[ii])
    {
      int32 pid = pid_ptr[ii];
      const T sample_val = sample_val_ptr[ii];
      int32 sample_idx = static_cast<int32>(sample_val * float32(color_map_size - 1));

      Vec4f sample_color = color_map_ptr[sample_idx];

      Vec<T,3> normal = normal_ptr[ii];
      Vec<T,3> hit_pt = hit_pt_ptr[ii];
      Vec<T,3> view_dir = -ray_dir_ptr[ii];
      
      Vec<T,3> light_dir = light_pos - hit_pt;
      light_dir.normalize();
      T diffuse = clamp(dot(light_dir, normal), T(0), T(1));

      Vec4f shaded_color;
      shaded_color[0] = light_amb[0];
      shaded_color[1] = light_amb[1];
      shaded_color[2] = light_amb[2];
      shaded_color[2] = sample_color[3];
      
      // add the diffuse component
      for(int32 c = 0; c < 3; ++c)
      {
        shaded_color[c] += diffuse * light_color[c] * sample_color[c];
      }

      Vec<T,3> half_vec = view_dir + light_dir;
      half_vec.normalize();
      float32 doth = clamp(dot(normal, half_vec), T(0), T(1));
      float32 intensity = pow(doth, spec_pow);

      // add the specular component
      for(int32 c = 0; c < 3; ++c)
      {
        shaded_color[c] += intensity * light_color[c] * sample_color[c];
      }

      Vec4f color = img_ptr[pid];
      //composite
      sample_color[3] *= (1.f - color[3]);
      color[0] = color[0] + sample_color[0] * sample_color[3];
      color[1] = color[1] + sample_color[1] * sample_color[3];
      color[2] = color[2] + sample_color[2] * sample_color[3];
      color[3] = sample_color[3] + color[3];
      img_ptr[pid] = color;
    }
  });
}//blend

void composite_bg(Array<Vec4f> &color_buffer,
                 Vec4f &bg_color)

{
  // avoid lambda capture issues
  Vec4f background = bg_color;
  Vec4f *img_ptr = color_buffer.get_device_ptr();
  const int32 size = color_buffer.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Vec4f color = img_ptr[i];
    if(color[3] < 1.f)
    {
      //composite
      float32 alpha = background[3] * (1.f - color[3]);
      color[0] = color[0] + background[0] * alpha;
      color[1] = color[1] + background[1] * alpha;
      color[2] = color[2] + background[2] * alpha;
      color[3] = alpha + color[3];
      img_ptr[i] = color;
    }
  });
}

} // namespace detail


//
// MeshField::integrate()
//
template <typename T>
Array<Vec<float32,4>>
MeshField<T>::integrate(Ray<T> rays, T sample_dist) const
{
  // set up a color table
  //ColorTable color_table("cool2warm");
  ColorTable color_table("Spectral");
  //color_table.add_alpha(0.f, 0.05f);
  //color_table.add_alpha(1.f, 0.05f);
  Array<Vec<float32, 4>> color_map;
  constexpr int color_samples = 1024;
  color_table.sample(color_samples, color_map);

  detail::calc_ray_start(rays, get_bounds());

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());
  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  Vec<float32,4> bg_color = make_vec4f(1.f,1.f,1.f,1.f);
  array_memset_vec(color_buffer, init_color);

  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.
  array_copy(rays.m_dist, rays.m_near);

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  //Array<int32> active_rays = array_counting(rays.size(),0,1);
  rays.m_active_rays = compact(rays.m_active_rays, rays.m_dist, rays.m_far, detail::IsLess<T>());

  int32 dbg_count_iter = 0;
  while(rays.m_active_rays.size() > 0) 
  {
    // Find elements and reference coordinates for the points.
    locate(rays.calc_tips(), rays.m_active_rays, rays.m_hit_idx, rays.m_hit_ref_pt);

    // Retrieve shading information at those points (scalar field value, gradient).
    ShadingContext<T> shading_ctx = get_shading_context(rays);

    // shade and blend sample using shading context  with color buffer
    detail::blend(color_buffer, color_map, shading_ctx);

    detail::advance_ray(rays, sample_dist); 

    rays.m_active_rays = compact(rays.m_active_rays, rays.m_dist, rays.m_far, detail::IsLess<T>());

    std::cout << "MeshField::integrate() - Finished iteration " << dbg_count_iter++ << std::endl;
  }
	detail::composite_bg(color_buffer,bg_color);
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


//
// MeshField::locate()
//
template<typename T>
void
MeshField<T>::locate(const Array<Vec<T,3>> points, Array<int32> &elt_ids, Array<Vec<T,3>> &ref_pts) const
{
  const Array<int32> active_idx = array_counting(points.size(), 0,1);
    // Assume that elt_ids and ref_pts are sized to same length as points.
  locate(points, active_idx, elt_ids, ref_pts);
}

//
// MeshField::locate()
//
template<typename T>
void
MeshField<T>::locate(const Array<Vec<T,3>> points, const Array<int32> active_idx, Array<int32> &elt_ids, Array<Vec<T,3>> &ref_pts) const
{
  using ShapeOpType = BShapeOp<ref_dim>;
  using TransOpType = ElTransOp<T, ShapeOpType, ElTransIter<T,space_dim> >;

  const int32 size = points.size();
  const int32 size_active = active_idx.size();
  const int32 size_aux = ShapeOpType::get_aux_req(m_p_space);
  const int32 el_dofs_space = m_eltrans_space.m_el_dofs;

  PointLocator locator(m_bvh);  
  constexpr int32 max_candidates = 5;
  Array<int32> candidates = locator.locate_candidates(points, active_idx, max_candidates);  //Size size_active * max_candidates.

  // For now the initial guess will always be the center of the element. TODO
  Vec<T,ref_dim> _ref_center;
  _ref_center = 0.5f;
  const Vec<T,ref_dim> ref_center = _ref_center;

  // Initialize outputs to well-defined dummy values.
  const Vec<T,3> three_point_one_four = {3.14, 3.14, 3.14};
  array_memset_vec(ref_pts, active_idx, three_point_one_four);
  array_memset(elt_ids, active_idx, -1);

    // Assume that elt_ids and ref_pts are sized to same length as points.

  // Auxiliary memory for evaluating element transformations.
  Array<T> aux_array;
  aux_array.resize(size_aux * size_active);

  const int32    *active_idx_ptr = active_idx.get_device_ptr_const();
  const Vec<T,3> *points_ptr     = points.get_device_ptr_const();
  const int32    *candidates_ptr = candidates.get_device_ptr_const();
  const int32    *ctrl_idx_ptr   = m_eltrans_space.m_ctrl_idx.get_device_ptr_const();
  const Vec<T,3> *ctrl_val_ptr   = m_eltrans_space.m_values.get_device_ptr_const();
  int32    *elt_ids_ptr   = elt_ids.get_device_ptr();
  Vec<T,3> *ref_pts_ptr   = ref_pts.get_device_ptr();
  T        *aux_array_ptr = aux_array.get_device_ptr();

  const int32 p_space = m_p_space;
  constexpr typename NewtonSolve<T>::SolveStatus not_converged = NewtonSolve<T>::NotConverged; // local

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 ii = active_idx_ptr[aii];
    const Vec<T,3> target_pt = points_ptr[ii];

    // - Use aii to index into candidates.
    // - Use ii to index into points, elt_ids, and ref_pts.

    int32 count = 0;
    int32 el_idx = candidates_ptr[aii*max_candidates + count]; 
    Vec<T,ref_dim> ref_pt = ref_center;

    T * const aux_mem_ptr = aux_array_ptr + aii * size_aux;
    TransOpType trans;
    trans.init_shape(p_space, aux_mem_ptr);

    bool found_inside = false;
    while(!found_inside && count < max_candidates && el_idx != -1)
    {
      trans.m_coeff_iter.init_iter(ctrl_idx_ptr, ctrl_val_ptr, el_dofs_space, el_idx);
      ref_pt = ref_center;    // Initial guess.

      const float32 tol_phys = 0.00001;      // TODO
      const float32 tol_ref  = 0.00001;

      int32 steps_taken;
      typename NewtonSolve<T>::SolveStatus status = not_converged;
      status = NewtonSolve<T>::solve(trans, target_pt, ref_pt, tol_phys, tol_ref, steps_taken);

      if ( status != not_converged && ShapeOpType::is_inside(ref_pt) )
      {
        // Found the element. Stop search, preserving count and el_idx.
        found_inside = true;
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

    // After testing each candidate, now record the result.
    if (found_inside)
    {
      elt_ids_ptr[ii] = el_idx;
      ref_pts_ptr[ii] = ref_pt;
    }
    else
    {
      elt_ids_ptr[ii] = -1;
    }

  });
}


//
// MeshField::get_shading_context()
//
// Returns shading context of size rays.
// This keeps image-bound buffers aligned with rays.
// For inactive rays, the is_valid flag is set to false.
//
template <typename T>
ShadingContext<T>
MeshField<T>::get_shading_context(Ray<T> &rays) const
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

  using ShapeOpType = BShapeOp<ref_dim>;
  using SpaceTransType = ElTransOp<T, ShapeOpType, ElTransIter<T,space_dim> >;
  using FieldTransType = ElTransOp<T, ShapeOpType, ElTransIter<T,field_dim> >;

  const int32 size_rays = rays.size();
  //const int32 size_active_rays = rays.m_active_rays.size();

  ShadingContext<T> shading_ctx;
  shading_ctx.resize(size_rays);

  // Adopt the fields (m_pixel_id) and (m_dir) from rays to intersection_ctx.
  shading_ctx.m_pixel_id = rays.m_pixel_id;
  shading_ctx.m_ray_dir = rays.m_dir;

  // Initialize all to a default state of "invalid."
  array_memset(shading_ctx.m_is_valid, static_cast<int32>(0));   // All are initialized to "invalid."

  // Initialize other outputs to well-defined dummy values.
  const Vec<T,3> one_two_three = {123., 123., 123.};
  array_memset_vec(shading_ctx.m_hit_pt, one_two_three);
  array_memset_vec(shading_ctx.m_normal, one_two_three);
  array_memset(shading_ctx.m_sample_val, static_cast<T>(-3.14));
  array_memset(shading_ctx.m_gradient_mag, static_cast<T>(55.55));
  
  // Refine the set of indices to which we must write valid states..
  Array<int32> active_valid_idx = compact(rays.m_active_rays, rays.m_hit_idx, detail::IsNonnegative<int32>());
  const int32 size_active_valid = active_valid_idx.size();

  const Range field_range = get_scalar_range();
  const T field_min = field_range.min();
  const T field_range_rcp = rcp_safe( field_range.length() );

  const int32 el_dofs_space = m_eltrans_space.m_el_dofs;
  const int32 el_dofs_field = m_eltrans_field.m_el_dofs;
  const int32 size_aux_space = SpaceTransType::get_aux_req(m_p_space);
  const int32 size_aux_field = FieldTransType::get_aux_req(m_p_field);
  const int32 size_aux = max(size_aux_space, size_aux_field);
  // Auxiliary memory to help evaluate element transformations.
  Array<T> aux_array;
  aux_array.resize(size_aux * size_active_valid);
  array_memset(aux_array, (T) -1.);   // Dummy value.

  const int32    *active_valid_ptr  = active_valid_idx.get_device_ptr_const();
  const Vec<T,3> *dir_ptr           = rays.m_dir.get_device_ptr_const();
  const Vec<T,3> *orig_ptr          = rays.m_orig.get_device_ptr_const();
  const T        *dist_ptr          = rays.m_dist.get_device_ptr_const();
  const int32    *hit_idx_ptr       = rays.m_hit_idx.get_device_ptr_const();
  const Vec<T,3> *hit_ref_pt_ptr    = rays.m_hit_ref_pt.get_device_ptr_const();

  const int32    *space_idx_ptr    = m_eltrans_space.m_ctrl_idx.get_device_ptr_const();
  const Vec<T,3> *space_val_ptr    = m_eltrans_space.m_values.get_device_ptr_const();
  const int32    *field_idx_ptr    = m_eltrans_field.m_ctrl_idx.get_device_ptr_const();
  const Vec<T,1> *field_val_ptr    = m_eltrans_field.m_values.get_device_ptr_const();
  T *aux_array_ptr = aux_array.get_device_ptr();

  int32    *is_valid_ptr      = shading_ctx.m_is_valid.get_device_ptr();
  Vec<T,3> *hit_pt_ptr        = shading_ctx.m_hit_pt.get_device_ptr();
  T        *sample_val_ptr    = shading_ctx.m_sample_val.get_device_ptr();
  Vec<T,3> *normal_ptr        = shading_ctx.m_normal.get_device_ptr();
  T        *gradient_mag_ptr  = shading_ctx.m_gradient_mag.get_device_ptr();

  const int32 p_space = m_p_space;  // local
  const int32 p_field = m_p_field;
  
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active_valid), [=] DRAY_LAMBDA (int32 aray_idx)
  {
    const int32 rii = active_valid_ptr[aray_idx];

    // Since we have filtered out all rays with (hit_idx == -1), all remaining rays are valid.
    is_valid_ptr[rii] = 1;

    // Compute hit point using ray origin, direction, and distance.
    hit_pt_ptr[rii] = orig_ptr[rii] + dir_ptr[rii] * dist_ptr[rii];

    // Evaluate element transformation to get scalar field value and gradient.

    const int32 el_id = hit_idx_ptr[rii];
    const Vec<T,3> ref_pt = hit_ref_pt_ptr[rii];
    T * const aux_mem_ptr = aux_array_ptr + aray_idx * size_aux;   // size_aux is big enough for either transformation.

    Vec<T,3> space_val;
    Vec<Vec<T,3>,3> space_deriv;
    {
      SpaceTransType trans_space;
      trans_space.init_shape(p_space, aux_mem_ptr);
      trans_space.m_coeff_iter.init_iter(space_idx_ptr, space_val_ptr, el_dofs_space, el_id);
      trans_space.eval(ref_pt, space_val, space_deriv);
    }

    Vec<T,1> field_val;
    Vec<Vec<T,1>,3> field_deriv;
    {
      FieldTransType trans_field;
      trans_field.init_shape(p_field, aux_mem_ptr);
      trans_field.m_coeff_iter.init_iter(field_idx_ptr, field_val_ptr, el_dofs_field, el_id);
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
    sample_val_ptr[rii] = (field_val[0] - field_min) * field_range_rcp;
    gradient_mag_ptr[rii] = gradient.magnitude();
    gradient.normalize();   //TODO What if the gradient is (0,0,0)?
    normal_ptr[rii] = gradient;
  });

  return shading_ctx;
}

// ---------------------------------------------
// Support for MeshField::intersect_isosurface()
// ---------------------------------------------

namespace detail
{
  template <typename T>
  void candidate_ray_intersection(Ray<T> rays, const BVH bvh)
  {
    Array<int32> old_active = rays.m_active_rays;
    Array<int32> temp_active;
    rays.m_active_rays = temp_active;
    rays.m_active_rays.resize( old_active.size() );
    array_copy(rays.m_active_rays, old_active);
  
    Array<int32> needs_test;       // -1 if doesn't need test, otherwise it does.
    needs_test.resize( rays.size() );
    array_memset(needs_test, 1);

    //TODO in order to flip this while-RAJA construction, need to factor out kernel of PointLocator::locate().
  
    const T sample_dist = 0.1;  // Dummy sample distance.. Should depend on mesh resolution (& curvature).
    while (rays.m_active_rays.size() > 0)
    {
      const int32 size_active = rays.m_active_rays.size();
  
      PointLocator locator(bvh);
      Array<int32> candidates = locator.locate_candidates(rays.calc_tips(), rays.m_active_rays, 1);  //Size size_active * max_candidates.

      const int32 *r_active_rays_ptr = rays.m_active_rays.get_device_ptr_const();
      const T *r_dist_ptr = rays.m_dist.get_device_ptr_const();
      const T *r_far_ptr = rays.m_far.get_device_ptr_const();
      const int32 *candidates_ptr = candidates.get_device_ptr_const();
  
      int32 *r_hit_idx_ptr = rays.m_hit_idx.get_device_ptr();
      int32 *needs_test_ptr = needs_test.get_device_ptr();
  
      // Fill in new information, and restrict the list of rays that still need to be found.
      RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
      {
        const int32 rii = r_active_rays_ptr[aii];
        r_hit_idx_ptr[rii] = candidates_ptr[aii];
        needs_test_ptr[rii] = ( (candidates_ptr[aii] == -1) && (r_dist_ptr[rii] < r_far_ptr[rii]) ) ? 1 : -1;
      });
      rays.m_active_rays = compact(rays.m_active_rays, needs_test, detail::IsNonnegative<int32>());
  
      // Advance the rays that still need to be found.
      detail::advance_ray(rays, sample_dist);
    }
  
    rays.m_active_rays = old_active;
  }
}  // namespace detail


// Make "static constexpr" work with some linkers.
template <typename T> constexpr int32 MeshField<T>::ref_dim;
template <typename T> constexpr int32 MeshField<T>::space_dim;
template <typename T> constexpr int32 MeshField<T>::field_dim;

// Explicit instantiations.
template class MeshField<float32>;
template class MeshField<float64>;




} // namespace dray

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

template <typename T, int32 RefDim, typename Shape1D>
void TensorShape<T, RefDim, Shape1D>::calc_shape_dshape(
    const Array<int32> &active_idx,
    const Array<Vec<T,RefDim>> &ref_pts,
    Array<T> &shape_val,                          // Will be resized.
    Array<Vec<T,RefDim>> &shape_deriv) const        // Will be resized.
{
  //
  // This method uses the virtual method el_dofs_1d().
  //

    // - Use size_queries, q_idx to index into ref_pts.
    // - Use size_active, a_q_idx to index into active_idx, shape_va, shape_deriv.
  const int32 size_queries = ref_pts.size();
  const int32 size_active = active_idx.size();
  const int32 el_dofs_1d = get_el_dofs_1d();      // Virtual method.
  const int32 el_dofs = get_el_dofs();            // Class method, but uses virtual get_el_dofs_1d().

  // Intermediate output arrays for 1D polynomials.
  Array<T> shape_val_1d;          // 1D eval for each ref variable, stored as XXXYYYZZZ|XXXYYYZZZ|...
  Array<T> shape_deriv_1d;        // 1D deriv eval for each ref variable, stored as XXXYYYZZZ|XXXYYYZZZ|...
  shape_val_1d.resize(el_dofs_1d * RefDim * size_active);
  shape_deriv_1d.resize(el_dofs_1d * RefDim * size_active);

  // Compute intermediate outputs.
  // The 1D polynomial for each variable can be evaluated at each ref_pt independently.
  T *val_1d_ptr = shape_val_1d.get_device_ptr();
  T *deriv_1d_ptr = shape_deriv_1d.get_device_ptr();
  const int32 *active_idx_ptr = active_idx.get_device_ptr_const();
  const Vec<T,RefDim> *ref_ptr = ref_pts.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, RefDim * size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 rdim = aii % RefDim;
    const int32 a_q_idx = aii / RefDim;
    const int32 q_idx = active_idx_ptr[a_q_idx];

    const T ref_coord = ref_ptr[q_idx][rdim];

    const int32 out_offset = el_dofs_1d * aii;
    Shape1D::calc_shape_dshape_1d(el_dofs_1d,
        ref_coord, 1-ref_coord,
        val_1d_ptr + out_offset,
        deriv_1d_ptr + out_offset);
  });

  // Create new output arrays shape_val and shape_deriv, of size_active.
  shape_val.resize(el_dofs * size_active);
  shape_deriv.resize(el_dofs * size_active);

  // Compute and store the tensor product of the 1D arrays.
  // Each component of the tensor product can be evaluated at each ref_pt independently.
  T *val_ptr = shape_val.get_device_ptr();
  Vec<T,RefDim> *deriv_ptr = shape_deriv.get_device_ptr();
  TensorProduct<T,RefDim,1,1> tensor_product_val;
  TensorProduct<T,RefDim,1,RefDim> tensor_product_deriv;
  
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, el_dofs * size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 tens_dim = aii % el_dofs;
    const int32 a_q_idx = aii / el_dofs;

    //TODO I think the bug is somewhere in this indexing madness.

    // Calculate the start of each 1D polynomial eval array for the current q_idx.
    const T* val_starts[RefDim];
    const T* deriv_starts[RefDim];
    for (int32 rdim = 0; rdim < RefDim; rdim++)
    {
      int32 offset_1d = el_dofs_1d * (rdim + RefDim * a_q_idx);
      val_starts[rdim] = val_1d_ptr + offset_1d;
      deriv_starts[rdim] = deriv_1d_ptr + offset_1d;
    }

    const int32 offset_q = el_dofs * a_q_idx;

    // Compute the output value by taking tensor product of all value rows.
    tensor_product_val(el_dofs_1d, val_starts, tens_dim, val_ptr + offset_q);

    // Compute each partial derivative by taking the tensor product
    // of that 1D derivative with the values of the other 1D rows.
    // Each partial derivative is a component in a Vec<RefDim>, so
    // the output stride is RefDim.
    for (int32 rdim = 0; rdim < RefDim; rdim++)
    {
      // Substitute 1D derivative in place of value, for the rdim- partial derivative.
      const T* const swap_val_start = val_starts[rdim];
      val_starts[rdim] = deriv_starts[rdim];

      T* const deriv_comp_ptr = (T *) (deriv_ptr + offset_q) + rdim;

      tensor_product_deriv(el_dofs_1d, val_starts, tens_dim, deriv_comp_ptr);

      // Undo the initial substitution.
      val_starts[rdim] = swap_val_start;
    }
  });
}

// Explicit instantiations.
template class TensorShape<float32, 3, Linear1D<float32>>;
template class TensorShape<float64, 3, Linear1D<float64>>;
template class TensorShape<float32, 3, Bernstein1D<float32>>;
template class TensorShape<float64, 3, Bernstein1D<float64>>;
template class TensorShape<float32, 2, Bernstein1D<float32>>;    //Debug
template class TensorShape<float64, 2, Bernstein1D<float64>>;    //Debug
template class TensorShape<float32, 1, Bernstein1D<float32>>;    //Debug
template class TensorShape<float64, 1, Bernstein1D<float64>>;    //Debug



  // Set the given attributes, and resize arrays.
template <typename T, int32 P, int32 R, typename ST>
void
ElTrans<T,P,R,ST>::resize(int32 size_el, int32 el_dofs, ST shape, int32 size_ctrl)
{
  // Check that el_dofs and RefDim are consistent with shape.
  assert( el_dofs == shape.get_el_dofs() );
  assert( RefDim == shape.get_ref_dim() );

  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_ctrl;
  m_shape = shape;
  
  m_ctrl_idx.resize(size_el * el_dofs);
  m_values.resize(size_ctrl);
}

  // This method assumes that output arrays are already the right size.
  // It does not resize or assign new arrays to output parameters.
template <typename T, int32 P, int32 R, typename ST>
void
ElTrans<T,P,R,ST>::eval(const Array<int> &active_idx,
            const Array<int32> &el_ids, const Array<Vec<T,R>> &ref_pts,
            Array<Vec<T,P>> &trans_val, Array<Matrix<T,P,R>> &trans_deriv) const
{
  const int32 size_queries = ref_pts.size();
  const int32 size_active = active_idx.size();

  // Evaluate shape at all active reference point.
  Array<T> shape_val;
  Array<Vec<T,RefDim>> shape_deriv;
  m_shape.calc_shape_dshape(active_idx, ref_pts, shape_val, shape_deriv);
    // Now shape_val and shape_deriv have been resized according to active_idx.

  // Intermediate data.
  const T *shape_val_ptr = shape_val.get_device_ptr_const();
  const Vec<T,RefDim> *shape_deriv_ptr = shape_deriv.get_device_ptr_const();
 
  // Member data.
  const Vec<T,PhysDim> *values_ptr = m_values.get_device_ptr_const();
  const int32 *ctrl_idx_ptr = m_ctrl_idx.get_device_ptr_const();
  const int32 el_dofs = m_el_dofs;   // local for lambda capture.

  // Output data.
  Vec<T,PhysDim> *trans_val_ptr = trans_val.get_device_ptr();
  Matrix<T,PhysDim,RefDim> *trans_deriv_ptr = trans_deriv.get_device_ptr();

  // Input data.
  const int32 *active_idx_ptr = active_idx.get_device_ptr_const();
  const int32 *el_ids_ptr = el_ids.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 q_idx = active_idx_ptr[aii];
    const int32 el_idx = el_ids_ptr[q_idx];

    // Grab and accumulate the control point values for this element,
    // weighted by the shape values.
    // (This part is sequential and not parallel because it is a "segmented reduction.")
    Vec<T,PhysDim> elt_val;
    elt_val = static_cast<T>(0.f);
    Matrix<T,PhysDim,RefDim> elt_deriv;
    elt_deriv = static_cast<T>(0.f);

    for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
    {
      // Get control point values.
      const int32 ci = ctrl_idx_ptr[el_idx * el_dofs + dof_idx];
      const Vec<T,PhysDim> cv = values_ptr[ci];

      // Shape values are weights for control point values -> element value.
      const T sv = shape_val_ptr[aii * el_dofs + dof_idx];
      elt_val += cv * sv;

      // Shape derivatives are weights for control point values -> element derivatives.
      const Vec<T,RefDim> sd = shape_deriv_ptr[aii * el_dofs + dof_idx];
      elt_deriv += Matrix<T,PhysDim,RefDim>::outer_product(cv, sd);
    }

    trans_val_ptr[q_idx] = elt_val;
    trans_deriv_ptr[q_idx] = elt_deriv;
  });
}

// Explicit instantiations.
template class ElTrans<float32, 1, 3, BernsteinShape<float32, 3>>;  //e.g. scalar field
template class ElTrans<float64, 1, 3, BernsteinShape<float64, 3>>;

template class ElTrans<float32, 3, 1, BernsteinShape<float32, 1>>;  //e.g. ray in space
template class ElTrans<float64, 3, 1, BernsteinShape<float64, 1>>;

template class ElTrans<float32, 3, 3, BernsteinShape<float32, 3>>;  //e.g. high-order geometry
template class ElTrans<float64, 3, 3, BernsteinShape<float64, 3>>;

template class ElTrans<float32, 4, 4, BernsteinShape<float32, 4>>;
template class ElTrans<float64, 4, 4, BernsteinShape<float64, 4>>;



// -----------------------
// Support for NewtonSolve
// -----------------------


//
// TODO  Combine RAJA kernels into a longer kernel.
//
// Currently the function to evaluate transformation functions contains its own RAJA loop.
// Hence the RAJA loops live inside the while loop (Newton Method iteration).
// The Newton Step algorithm has to be split into separate kernels on either side of that synchronization.
// Or, we have to do weird loop counter tricks.
//
// In the future, the evaluation code will be made available in a kernel-executable form.
// Once that happens, each (Newton) query will be independent of the others.
// Kernels can be combined. Each query can iterate several times in its own thread.
// That is, the while loop can live inside the RAJA loop. The only reason to synchronize threads
// midway into the solve would be to compact the "active" threads.
//
//
// ** How the Loop Should Be Structured **
// 
// RAJA::forall {
//   indices = get_indices();
//   Vec<Ref> new_ref = initial_guess[idx];
//   evaluate_transformation(shared_mem_ptr, indices, new_ref, (out) y, (out) deriv);
//   delta_y = target_ptr[idx] - y;
//   convergence_status = (delta_y.norm < tol_phys) ? ConvergePhys : NotConverged;
//
//   steps_taken = 0;
//   while (steps_taken < max_steps && convergence_status == NotConverged)
//   {
//     is_valid = Matrix::solve_y_equals_A_x(delta_y, deriv, (out) delta_x);
//     if (is_valid)
//     {
//       // If converged, we're done.
//       convergence_status = (delta_x.norm < tol_ref) ? ConvergeRef : NotConverged;
//       if (convergence_status == ConvergeRef)
//         break;
//
//       // Otherwise, apply the Newton increment.
//       new_ref = new_ref + delta_x;
//       steps_taken++;
//     }
//     else
//     {
//       // Uh-oh. Some kind of singularity.
//       break;
//     }
//
//     evaluate_transformation(shared_mem_ptr, indices, new_ref, (out) y, (out) deriv);
//     delta_y = target_ptr[idx] - y;
//     convergence_status = (delta_y.norm < tol_phys) ? ConvergePhys : NotConverged;
//   }  // end while
//
//   if (steps_taken > 0)
//   {
//     set_ref(ref_ptr, idx, new_ref);
//   }
// } // end RAJA
//
//

template <typename QueryType>
int32 NewtonSolve<QueryType>::step(
    const Array<Vec<T,phys_dim>> &target,
    QueryType &q,
    const Array<int32> &query_active,
    Array<int32> &solve_status,
    int32 max_steps)
{
  //TODO make these as parameters somewhere
  constexpr T tol_phys = 0.0000001;
  constexpr T tol_ref = 0.0000001;


  const int32 size_query = q.size();
  const int32 size_active = query_active.size();

  solve_status.resize(size_active);   // Resize output.
  array_memset(solve_status, (int32) NotConverged);

  // Device pointers.
  int32 *solve_status_ptr = solve_status.get_device_ptr();
  const int32 *active_idx_ptr = query_active.get_device_ptr_const();
  const Vec<T,phys_dim> *target_ptr = target.get_device_ptr_const();

  typename QueryType::ptr_bundle_const_t val_ptrb = q.get_val_device_ptr_const();
  typename QueryType::ptr_bundle_const_t deriv_ptrb = q.get_deriv_device_ptr_const();
  typename QueryType::ptr_bundle_t ref_ptrb = q.get_ref_device_ptr();

  // The initial guess for each query is already loaded in query.m_ref_pts.

  int32 num_not_convg = size_active;

  // One or more Newton steps.
  ///int32 steps_taken = 0;
  ///while (steps_taken < max_steps && num_not_convg > 0)

  // Workaround for kernel division (see long note above). Instead of counting Newton steps,
  // we will count transformation-evaluations and physical-convergence-assessments.
  //
  int32 eval_count = 0;
  while (eval_count <= max_steps && num_not_convg > 0)
  {
    // Compute the physical images of initial reference points, and test for convergence.
    q.query(query_active);
    
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
    {
      const int32 q_idx = active_idx_ptr[aii];

      Vec<T,ref_dim> delta_y;

      // Assess physical convergence.
      if (solve_status_ptr[aii] == NotConverged)
      {
        // Compute delta_y.
        delta_y = QueryType::get_val(val_ptrb, q_idx);
        delta_y = target_ptr[q_idx] - delta_y;

        // Optionally check boundary -- currently not.

        // Check for convergence in physical coordinates.
        solve_status_ptr[aii] = (delta_y.Normlinf() < tol_phys) ? ConvergePhys : NotConverged;
      }
      const int32 phys_assess_count = eval_count + 1;

      // Apply Newton's method to get delta_x, and assess reference convergence.
      if (phys_assess_count <= max_steps && solve_status_ptr[aii] == NotConverged)
      {
        Vec<T,ref_dim> delta_x;
        bool inverse_valid;
        Matrix<T,phys_dim,ref_dim> jacobian = QueryType::get_deriv(deriv_ptrb, q_idx);
        delta_x = matrix_mult_inv(jacobian, delta_y, inverse_valid);  //Compiler error if ref_dim != phys_dim.

        // Check for convergence in reference coordinates
        int32 status = (inverse_valid && delta_x.Normlinf() < tol_ref) ? ConvergeRef : NotConverged;
        solve_status_ptr[aii] = status;

        // If need to continue iterating, then update the reference coordinate by adding delta_x.
        if (inverse_valid && status == NotConverged)
        {
          Vec<T,ref_dim> new_ref = QueryType::get_ref(ref_ptrb, q_idx) + delta_x;
          QueryType::set_ref(ref_ptrb, q_idx, new_ref);
        }
      }
    });  // end RAJA Newton Step

    RAJA::ReduceSum<reduce_policy, int32> raja_num_not_convg(0);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
    {
      raja_num_not_convg += (solve_status_ptr[aii] == NotConverged) ? 1 : 0;
    });
    num_not_convg = raja_num_not_convg.get();

    eval_count++;
  }  // end while loop.

  int32 num_steps = eval_count - 1;
  return num_steps;
}

// Explicit instantiations.
///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 1, 3>>>;  //e.g. scalar field
///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 1, 3>>>;

///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 3, 1>>>;  //e.g. ray in space
///template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 3, 1>>>;

template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 3, 3>>>;  //e.g. high-order geometry
template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 3, 3>>>;

template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float32, 4, 4>>>;
template class NewtonSolve<ElTransQuery<ElTrans_BernsteinShape<float64, 4, 4>>>;


// -----------------------
// Support for MeshField
// -----------------------

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
  const Vec4f *color_map_ptr = color_map.get_device_ptr_const();

  Vec4f *img_ptr = color_buffer.get_device_ptr();

  const int color_map_size = color_map.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, shading_ctx.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    if (is_valid_ptr[ii])
    {
      int32 pid = pid_ptr[ii];
      const T sample_val = sample_val_ptr[ii];
      int32 sample_idx = static_cast<int32>(sample_val * float32(color_map_size - 1));
      Vec4f sample_color = color_map_ptr[sample_idx];

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
}

} // namespace detail


//
// MeshField::integrate()
//
template <typename T, class ETS, class ETF>
Array<Vec<float32,4>>
MeshField<T,ETS,ETF>::integrate(Ray<T> rays, T sample_dist)
{
  // set up a color table
  ColorTable color_table("cool2warm");
  color_table.add_alpha(0.f, 0.1f);
  color_table.add_alpha(1.f, 0.1f);
  Array<Vec<float32, 4>> color_map;
  constexpr int color_samples = 1024;
  color_table.sample(color_samples, color_map);

  detail::calc_ray_start(rays, get_bounds());

  // Initialize the color buffer to (0,0,0,0).
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize(rays.size());
  Vec<float32,4> init_color = make_vec4f(0.f,0.f,0.f,0.f);
  array_memset_vec(color_buffer, init_color);

  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.
  array_copy(rays.m_dist, rays.m_near);

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  //Array<int32> active_rays = array_counting(rays.size(),0,1);
  rays.m_active_rays = compact(rays.m_active_rays, rays.m_dist, rays.m_far, detail::IsLess<T>());

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
  }

  return color_buffer;
}


//
// MeshField::make_bvh()
//
template <typename T, class ETS, class ETF>
void MeshField<T,ETS,ETF>::make_bvh()
{
  constexpr double bbox_scale = 1.000001;

  const int num_els = m_size_el;
  const int32 el_dofs = m_eltrans_space.get_el_dofs();
  
  Array<AABB> aabbs;
  aabbs.resize(num_els); 
  AABB *aabb_ptr = aabbs.get_device_ptr();

  const int32 *ctrl_idx_ptr = m_eltrans_space.get_m_ctrl_idx_const().get_device_ptr_const();
  const Vec<T,space_dim> *ctrl_val_ptr = m_eltrans_space.get_m_values_const().get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 elem)
  {
    AABB bbox;
  
    // Add each dof of the element to the bbox
    // Note: positivity of Bernstein bases ensures that convex
    //       hull of element nodes contain entire element
    for (int32 dof = 0; dof < el_dofs; dof++)
    {
      const int32 cidx = ctrl_idx_ptr[elem * el_dofs + dof];
      const Vec<T, space_dim> &v = ctrl_val_ptr[cidx];
      const Vec3f v3f = {(float32) v[0], (float32) v[1], (float32) v[2]};
      bbox.include(v3f);
    }
 
    // Slightly scale the bbox to account for numerical noise
    bbox.scale(bbox_scale);

    aabb_ptr[elem] = bbox;
  });

  LinearBVHBuilder builder;
  m_bvh = builder.construct(aabbs);
}


//
// MeshField::locate()
//
template<typename T, class ETS, class ETF>
void
MeshField<T,ETS,ETF>::locate(const Array<Vec<T,3>> points, Array<int32> &elt_ids, Array<Vec<T,3>> &ref_pts)
{
  const Array<int32> active_idx = array_counting(points.size(), 0,1);
    // Assume that elt_ids and ref_pts are sized to same length as points.
  locate(points, active_idx, elt_ids, ref_pts);
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


//
// MeshField::locate()
//
template<typename T, class ETS, class ETF>
void
MeshField<T,ETS,ETF>::locate(const Array<Vec<T,3>> points, const Array<int32> active_idx, Array<int32> &elt_ids, Array<Vec<T,3>> &ref_pts)
{
  // Continue the workaround since NewtonSolve currently contains its own RAJA loop.
  // Therefore when we iterate over candidates, we must loop outside of RAJA.
  //
  // For now:
  //
  // intern_active_idx = copy(active_idx);
  //
  // space_query;
  //
  // cand_idx = 0;
  // set(space_query.m_el_ids, candidates, cand_idx, intern_active_idx)
  // set(space_query.m_ref_pts, init_guess(points, guess_type), intern_active_idx);
  // intern_active_idx = compact(ref_pts, el_ids, intern_active_idx, solve_status, HasCandidate);
  //
  // while (cand_idx < max_candidates && intern_active_idx.size() > 0)
  // {
  //   size_active = intern_active_idx.size();
  //   NewtonSolve::step(points, query, intern_active_idx, solve_status);
  //   set(space_query.m_el_ids, candidates, cand_idx, intern_active_idx)
  //   set(space_query.m_ref_pts, init_guess(points, guess_type), intern_active_idx);
  //   intern_active_idx = compact(ref_pts, el_ids, intern_active_idx, solve_status, HasCandidate && !IsInside);
  // }

  const int32 size_points = points.size();
  const int32 size_active = active_idx.size();

  // Initialize outputs to well-defined dummy values.
  const Vec<T,3> three_point_one_four = {3.14, 3.14, 3.14};
  array_memset_vec(ref_pts, active_idx, three_point_one_four);
  array_memset(elt_ids, active_idx, -1);

  // For now the initial guess will always be the center of the element. TODO
  Vec<T,ref_dim> _ref_center;
  _ref_center = 0.5f;
  const Vec<T,ref_dim> ref_center = _ref_center;

  typedef ElTransQuery<ETS> QType;
  typedef NewtonSolve<QType> NSType;
  QType space_query;
    // Assume that elt_ids and ref_pts are sized to same length as points.
  space_query.m_el_ids = elt_ids;
  space_query.m_ref_pts = ref_pts;
  space_query.m_result_val = points;
  space_query.m_result_deriv.resize( points.size() );  // Unused, but still needed.

  // Get candidate element ids.
  PointLocator locator(m_bvh);  
  constexpr int32 max_candidates = 5;
  Array<int32> candidates = locator.locate_candidates(points, active_idx, max_candidates);  //Size active_size * max_candidates.
  const int32 *candidates_ptr = candidates.get_device_ptr_const();

  // Since NewtonSolve contains its own RAJA loop, we our outer loop will iterate over candidates.
  // Not all elements have the same number of candidates. Not all elements find the inside in the
  // same number of attempts. Therefore, to avoid testing nonexistent or already-found candidates,
  // we'll compact the list of un-found points on every iteration of the outer loop.
  // We also need a map from that gets us to index space relative to active_idx.
  Array<int32> searchable_to_active;
  Array<int32> searchable_idx;
  searchable_to_active = array_counting( active_idx.size(), 0,1);
  searchable_idx = gather(active_idx, searchable_to_active);

  int32 cand_idx = 0;

  // Persistent device pointers.
  int32 *el_ids_ptr = space_query.m_el_ids.get_device_ptr();
  Vec<T,ref_dim> *ref_pts_ptr = space_query.m_ref_pts.get_device_ptr();

  // Set the element ids and reference points for the current cand_idx.
  int32 size_searchable = searchable_idx.size();
  const int32 *searchable_idx_ptr = searchable_idx.get_device_ptr_const();
  const int32 *searchable_to_active_ptr = searchable_to_active.get_device_ptr_const();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_searchable), [=] DRAY_LAMBDA (int32 sii)
  {
    const int32 aii = searchable_to_active_ptr[sii];
    const int32 qidx = searchable_idx_ptr[sii];
    el_ids_ptr[qidx] = candidates_ptr[aii * max_candidates + cand_idx];

    // For now always guess the center of the element. TODO
    ref_pts_ptr[qidx] = ref_center;
  });

  // Compaction.
    // In order to compact searchable_to_active, need an array of candidates
    // that is the same size as active_idx.
  Array<int32> current_candidates = gather(candidates, array_counting(size_active, cand_idx, max_candidates));
  searchable_to_active = compact(searchable_to_active, current_candidates, IsNonnegative<int32>());
  searchable_idx = gather(active_idx, searchable_to_active);
  size_searchable = searchable_idx.size();

  // Ternary functor needed for compaction.
  //typedef IsSearchable<typename ETS::ShapeType, Vec<T,ref_dim>, int32, typename NSType::SolveStatus> is_searchable_t;
  typedef IsSearchable<typename ETS::ShapeType, Vec<T,ref_dim>, int32, int32> is_searchable_t;
  is_searchable_t is_searchable;

  while (cand_idx < max_candidates && searchable_idx.size() > 0)
  {
    Array<int32> solve_status;
    NewtonSolve<QType>::step(points, space_query, searchable_idx, solve_status);
      // Size of solve_status is now size_searchable.

    // Set the element ids and reference points for the current cand_idx.
    const int32 *searchable_idx_ptr = searchable_idx.get_device_ptr_const();
    const int32 *searchable_to_active_ptr = searchable_to_active.get_device_ptr_const();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_searchable), [=] DRAY_LAMBDA (int32 sii)
    {
      const int32 aii = searchable_to_active_ptr[sii];
      const int32 qidx = searchable_idx_ptr[sii];
      el_ids_ptr[qidx] = candidates_ptr[aii * max_candidates + cand_idx];
  
      // For now always guess the center of the element. TODO
      ref_pts_ptr[qidx] = ref_center;
    });

    // Compaction.
    current_candidates = gather(candidates, array_counting(size_active, cand_idx, max_candidates));

    searchable_to_active = compact<int32, Vec<T,ref_dim>, int32, int32, is_searchable_t>(
        searchable_idx, searchable_to_active,
        space_query.m_ref_pts, current_candidates, solve_status,
        is_searchable);

    searchable_idx = gather(active_idx, searchable_to_active);
    size_searchable = searchable_idx.size();

    cand_idx++;
  }

  if (searchable_idx.size() > 0)
  {
    std::cout << "MeshField::locate() - Warning: Search stopped but not all candidates were exhausted." << std::endl;
  }

  // Since we identified space_query.m_ref_pts with [out] ref_pts,
  // and space_query.m_el_ids with [out] elt_ids,
  // the output should be set now.
}
/// 
/// 
/// 
/// /*
///  * Returns shading context of size rays.
///  * This keeps image-bound buffers aligned with rays.
///  * For inactive rays, the is_valid flag is set to false.
///  */
/// template <typename T, class ETS, class ETF>
/// ShadingContext<T>
/// MeshField<T,ETS,ETF>::get_shading_context(Ray<T> &rays) const
/// {
///   const int32 size_rays = rays.size();
///   const int32 size_active_rays = rays.m_active_rays.size();
/// 
///   ShadingContext<T> shading_ctx;
///   shading_ctx.resize(size_rays);
/// 
///   // Initialize outputs to well-defined dummy values (except m_pixel_id and m_ray_dir; see below).
///   const Vec<T,3> one_two_three = {123., 123., 123.};
///   array_memset_vec(shading_ctx.m_hit_pt, one_two_three);
///   array_memset_vec(shading_ctx.m_normal, one_two_three);
///   array_memset(shading_ctx.m_sample_val, static_cast<T>(-3.14));
///   array_memset(shading_ctx.m_is_valid, static_cast<int32>(0));   // All are initialized to "invalid."
///   
///   // Adopt the fields (m_pixel_id) and (m_dir) from rays to intersection_ctx.
///   shading_ctx.m_pixel_id = rays.m_pixel_id, rays.m_active_rays;
///   shading_ctx.m_ray_dir = rays.m_dir, rays.m_active_rays;
/// 
///   // TODO cache this in a field of MFEMGridFunction.
///   T field_min, field_max;
///   field_bounds(field_min, field_max);
///   T field_range_rcp = rcp_safe(field_max - field_min);
/// 
///   const int32 *hit_idx_ptr = rays.m_hit_idx.get_host_ptr_const();
///   const Vec<T,3> *hit_ref_pt_ptr = rays.m_hit_ref_pt.get_host_ptr_const();
/// 
///   int32 *is_valid_ptr = shading_ctx.m_is_valid.get_host_ptr();
///   T *sample_val_ptr = shading_ctx.m_sample_val.get_host_ptr();
///   Vec<T,3> *normal_ptr = shading_ctx.m_normal.get_host_ptr();
///   //Vec<T,3> *hit_pt_ptr = shading_ctx.m_hit_pt.get_host_ptr();
/// 
///   const int32 *active_rays_ptr = rays.m_active_rays.get_host_ptr_const();
/// 
///   RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, size_active_rays), [=] (int32 aray_idx)
///   {
///     const int32 ray_idx = active_rays_ptr[aray_idx];
/// 
///     if (hit_idx_ptr[ray_idx] == -1)
///     {
///       // Sample is not in an element.
///       is_valid_ptr[ray_idx] = 0;
///     }
///     else
///     {
///       // Sample is in an element.
///       is_valid_ptr[ray_idx] = 1;
/// 
///       const int32 elt_id = hit_idx_ptr[ray_idx];
/// 
///       // Convert hit_ref_pt to double[3], then to mfem::IntegrationPoint..
///       double ref_pt[3];
///       ref_pt[0] = hit_ref_pt_ptr[ray_idx][0];
///       ref_pt[1] = hit_ref_pt_ptr[ray_idx][1];
///       ref_pt[2] = hit_ref_pt_ptr[ray_idx][2];
/// 
///       mfem::IntegrationPoint ip;
///       ip.Set(ref_pt, 3);
/// 
///       // Get scalar field value and copy to output.
///       const T field_val = m_pos_nodes->GetValue(elt_id, ip);
///       sample_val_ptr[ray_idx] = (field_val - field_min) * field_range_rcp;
/// 
///       // Get gradient vector of scalar field.
///       mfem::FiniteElementSpace *fe_space = GetGridFunction()->FESpace();
///       mfem::IsoparametricTransformation elt_trans;
///       //TODO Follow up: I wish there were a const method for this.
///       //I purposely used the (int, IsoparametricTransformation *) form to avoid mesh caching.
///       fe_space->GetElementTransformation(elt_id, &elt_trans);
///       elt_trans.SetIntPoint(&ip);
///       mfem::Vector grad_vec;
///       m_pos_nodes->GetGradient(elt_trans, grad_vec);
///       
///       // Normalize gradient vector and copy to output.
///       Vec<T,3> gradient = {static_cast<T>(grad_vec[0]),
///                            static_cast<T>(grad_vec[1]),
///                            static_cast<T>(grad_vec[2])};
///       T gradient_mag = gradient.magnitude();
///       gradient.normalize();   //TODO What if the gradient is (0,0,0)?
///       normal_ptr[ray_idx] = gradient;
/// 
///       //TODO store the magnitude of the gradient if that is desired.
/// 
///       //TODO compute hit point using ray origin, direction, and distance.
///     }
///   });
/// 
///   return shading_ctx;
/// }





// Explicit instantiations.
template class MeshField<float32, ElTrans_BernsteinShape<float32, 3,3>, ElTrans_BernsteinShape<float32, 1,3>>;
template class MeshField<float64, ElTrans_BernsteinShape<float64, 3,3>, ElTrans_BernsteinShape<float64, 1,3>>;


} // namespace dray

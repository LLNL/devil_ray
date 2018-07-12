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
  constexpr T tol_phys = 0.00001;
  constexpr T tol_ref = 0.00001;


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

  /// const int32 dbg_track_id = 31350;    // An active id reported by runtime.

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
    //RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, size_active), [=] (int32 aii)    // DEBUG
    {
      const int32 q_idx = active_idx_ptr[aii];

      /// //DEBUG
      /// if (q_idx == dbg_track_id)
      /// {
      ///   std::cout << "step " << eval_count << " |    ";
      ///   std::cout << "ref: " << QueryType::get_ref(ref_ptrb, q_idx) << "   |   ";
      ///   std::cout << "phys: " << QueryType::get_val(val_ptrb, q_idx);
      /// }

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

        /// //DEBUG
        /// if (q_idx == dbg_track_id)
        /// {
        ///   std::cout << "     | delta_y: " << delta_y << "   -> " << (inverse_valid ? "inv valid :)" : "inv INVALID") << ", delta_x: " << delta_x;
        ///   if (!inverse_valid)
        ///   {
        ///     std::cout << std::endl;
        ///     std::cout << jacobian;
        ///   }
        /// }

        // If need to continue iterating, then update the reference coordinate by adding delta_x.
        if (inverse_valid && status == NotConverged)
        {
          Vec<T,ref_dim> new_ref = QueryType::get_ref(ref_ptrb, q_idx) + delta_x;
          QueryType::set_ref(ref_ptrb, q_idx, new_ref);
        }
      }

      /// //DEBUG
      /// if (q_idx == dbg_track_id)
      /// {
      ///   std::cout << std::endl;
      /// }

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

  return color_buffer;
}


//
// MeshField::construct_bvh()
//
template <typename T, class ETS, class ETF>
BVH MeshField<T,ETS,ETF>::construct_bvh()
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
      const Vec<T,space_dim> &v = ctrl_val_ptr[cidx];
      const Vec3f v3f = {(float32) v[0], (float32) v[1], (float32) v[2]};
      bbox.include(v3f);
    }
 
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
template <typename T, class ETS, class ETF>
void MeshField<T,ETS,ETF>::field_bounds(T &field_min, T &field_max) const // TODO move this capability into the bvh structure.
{
  // The idea is...
  // First assume that we have a positive basis.
  // Then the global maximum and minimum are guaranteed to be found on nodes/vertices.

  RAJA::ReduceMin<reduce_policy, T> comp_min(infinity32());
  RAJA::ReduceMax<reduce_policy, T> comp_max(neg_infinity32());

  const int32 num_nodes = m_eltrans_field.get_m_values_const().size();
  const T *node_val_ptr = (const T*) m_eltrans_field.get_m_values_const().get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_nodes), [=] DRAY_LAMBDA (int32 ii)
  {
    comp_min.min(node_val_ptr[ii]);
    comp_max.max(node_val_ptr[ii]);
  });

  field_min = comp_min.get();
  field_max = comp_max.get();
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
  space_query.m_eltrans = m_eltrans_space;
    // Assume that elt_ids and ref_pts are sized to same length as points.
  space_query.resize(size_points);
  space_query.m_ref_pts = ref_pts;                // Shallow copy, same internal memory.

  // Get candidate element ids.
  PointLocator locator(m_bvh);  
  constexpr int32 max_candidates = 5;
  Array<int32> candidates = locator.locate_candidates(points, active_idx, max_candidates);  //Size active_size * max_candidates.
  ///std::cout << "Candidates    "; candidates.summary();  //DEBUG
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
  int32 *out_el_ids_ptr = elt_ids.get_device_ptr();
  int32 *q_el_ids_ptr = space_query.m_el_ids.get_device_ptr();
  Vec<T,ref_dim> *q_ref_pts_ptr = space_query.m_ref_pts.get_device_ptr();

  // Compaction.
    // In order to compact searchable_to_active, need an array of candidates
    // that is the same size as active_idx.
  Array<int32> candidate_lookahead = gather(candidates, array_counting(size_active, cand_idx, max_candidates));
  searchable_to_active = compact(searchable_to_active, candidate_lookahead, IsNonnegative<int32>());
  searchable_idx = gather(active_idx, searchable_to_active);
  int32 size_searchable = searchable_idx.size();

  // Ternary functor needed for compaction.
  //typedef IsSearchable<typename ETS::ShapeType, Vec<T,ref_dim>, int32, typename NSType::SolveStatus> is_searchable_t;
  typedef IsSearchable<typename ETS::ShapeType, Vec<T,ref_dim>, int32, int32> is_searchable_t;
  is_searchable_t is_searchable;

  Array<int32> solve_status;

  // Use Newton's Method on each spread of candidates until no point is unfound.
  while (cand_idx < max_candidates && searchable_idx.size() > 0)
  {
    // Init for NewtonSolve: Set the element ids and reference points for the current cand_idx.
    const int32 *searchable_idx_ptr = searchable_idx.get_device_ptr_const();
    const int32 *searchable_to_active_ptr = searchable_to_active.get_device_ptr_const();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_searchable), [=] DRAY_LAMBDA (int32 sii)
    {
      const int32 aii = searchable_to_active_ptr[sii];
      const int32 qidx = searchable_idx_ptr[sii];
      q_el_ids_ptr[qidx] = candidates_ptr[aii * max_candidates + cand_idx];
  
      // For now always guess the center of the element. TODO
      q_ref_pts_ptr[qidx] = ref_center;
    });

    // Newton Solve.
    //Array<int32> solve_status;
    int32 num_steps = NewtonSolve<QType>::step(points, space_query, searchable_idx, solve_status);
      // Size of solve_status is now size_searchable.

    // Set the output el_ids for the successful solves.
    const int32 *solve_status_ptr = solve_status.get_device_ptr_const();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_searchable), [=] DRAY_LAMBDA (int32 sii)
    {
      const int32 qidx = searchable_idx_ptr[sii];
      if (solve_status_ptr[sii] != NewtonSolve<QType>::NotConverged && ETS::ShapeType::IsInside(q_ref_pts_ptr[qidx]))
      {
        out_el_ids_ptr[qidx] = q_el_ids_ptr[qidx];
      }
    });

    // Compaction. If the compaction results in size_searchable==0, the loop will break.
    candidate_lookahead = gather(candidates, array_counting(size_active, cand_idx+1, max_candidates));

    searchable_to_active = compact<int32, Vec<T,ref_dim>, int32, int32, is_searchable_t>(
        searchable_idx, searchable_to_active,
        space_query.m_ref_pts, candidate_lookahead, solve_status,
        is_searchable);

    searchable_idx = gather(active_idx, searchable_to_active);
    size_searchable = searchable_idx.size();

    cand_idx++;
  }

  ///printf("Tried %d candidates (outer loop)\n", cand_idx);

  ///std::cout << "Final state of solve_status:  " << std::endl;
  ///solve_status.summary();

  if (searchable_idx.size() > 0)
  {
    std::cout << "MeshField::locate() - Warning: Search stopped but not all candidates were exhausted." << std::endl;
  }

  // We identified space_query.m_ref_pts with [out] ref_pts, and space_query.m_el_ids with [out] elt_ids.
  // It was a shallow copy, so the side effects of NewtonSolve are now instilled in the output parameters.
}


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
  
    const T sample_dist = 0.1;  // Dummy sample distance.. Should depend on mesh resolution (& curvature).
    while (rays.m_active_rays.size() > 0)
    {
      const int32 size_active = rays.m_active_rays.size();
  
      PointLocator locator(bvh);
      Array<int32> candidates = locator.locate_candidates(rays.calc_tips(), rays.m_active_rays, 1);  //Size active_size * max_candidates.

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
      rays.m_active_rays = compact(rays.m_active_rays, needs_test, IsNonnegative<int32>());
  
      // Advance the rays that still need to be found.
      detail::advance_ray(rays, sample_dist);
    }
  
    rays.m_active_rays = old_active;
  }
}  // namespace detail


//
// MeshField::intersect_isosurface()
//
template <typename T, class ETS, class ETF>
void MeshField<T,ETS,ETF>::intersect_isosurface(Ray<T> rays, T isoval) const
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

  constexpr int32 space_dim = MeshField::space_dim;  // Local for lambda captures.
  constexpr int32 field_dim = MeshField::field_dim;  // Local for lambda captures.
  constexpr int32 ref_dim = MeshField::ref_dim;      // Local for lambda captures.

  // TODO Here I have assumed that the spatial and scalar fields have BernsteinShape of some dimensions.
  // Not sure if there is a good way to identify the given (common) shape and use that with the combined dimensions.

  // Set up containers.
  typedef ElTrans_BernsteinShape<T, space_dim + field_dim, ref_dim> ETMeshField;
  typedef ElTrans_BernsteinShape<T, space_dim + field_dim, 1> ETRay;
  typedef ElTransQuery2<ETMeshField, ETRay> QMeshFieldRay;

  QMeshFieldRay intersection_system;
  ElTransQuery<ETMeshField> &q_meshfield_ref = intersection_system.m_q1;
  ElTransQuery<ETRay> &q_ray_ref = intersection_system.m_q2;

  Array<Vec<T,space_dim + field_dim>> target;

  //
  // Insert the input parameters:
  // - Size
  // - Ray initial guess
  // - Ray "field" data
  // - Ray "field" target
  //
  const int32 size_rays = rays.size();
  const int32 size_a_rays = rays.m_active_rays.size();

  intersection_system.resize(size_rays);
  target.resize(size_rays);

  // Insert magical initial guesses here.   TODO
  //
  // For now, choose a sample distance, and sample along each ray until enter bbox for some element.
  // Then, guess the center of the element.
  detail::calc_ray_start(rays, get_bounds());
  array_copy(rays.m_dist, rays.m_near);
  detail::candidate_ray_intersection<T>(rays, m_bvh);
  q_meshfield_ref.m_el_ids = rays.m_hit_idx;     // Element ids guesses go into q_meshfield_ref.
  //TODO check if array sharing is what we want here.

  // Only query if we have at least one candidate element.
  Array<int32> active_queries = compact(rays.m_active_rays, q_meshfield_ref.m_el_ids, IsNonnegative<int32>());
  const int32 size_a_queries = active_queries.size();

  Vec<T,ref_dim> _ref_center;
  _ref_center = 0.5;
  const Vec<T,ref_dim> &ref_center = _ref_center;
  array_memset_vec(q_meshfield_ref.m_ref_pts, ref_center);  // Element ref points are set to center.

  // Construct one field "element" per active ray.
  // Linear elements have polynomial degree 1 with two degrees of freedom.
  // Each ray has one unique dof (ray.m_dir) and one shared dof (ray.m_orig - ray.m_orig == 0).
  // The value of the shared dof is tacked to the end of the values array.
  q_ray_ref.m_eltrans.resize(size_a_queries, 2, BernsteinShape<T,1>::factory(1), size_a_queries + 1);

  { //scope
    const int32 *active_idx_ptr = active_queries.get_device_ptr_const();
    const Vec<T,space_dim> *r_dir_ptr = rays.m_dir.get_device_ptr_const();
    const Vec<T,space_dim> *r_orig_ptr = rays.m_orig.get_device_ptr_const();
    const T *r_dist_ptr = rays.m_dist.get_device_ptr_const();
    Vec<T,space_dim + field_dim> *q_ray_values_ptr = q_ray_ref.m_eltrans.get_m_values().get_device_ptr();
    int32 *q_ray_ctrl_idx_ptr = q_ray_ref.m_eltrans.get_m_ctrl_idx().get_device_ptr();
    int32 *q_ray_el_id_ptr = q_ray_ref.m_el_ids.get_device_ptr();
    Vec<T,1> *q_ray_ref_pt_ptr = q_ray_ref.m_ref_pts.get_device_ptr();
    Vec<T,space_dim + field_dim> *target_ptr = target.get_device_ptr();

    // Iterate over all active rays/active queries.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_a_queries+1), [=] DRAY_LAMBDA (int32 aii)
    {
      if (aii == size_a_queries)
      {
        q_ray_values_ptr[size_a_queries] = 0;
      }
      else
      {
        const int32 rii = active_idx_ptr[aii];
          // Control point value.
        for (int32 sdim = 0; sdim < space_dim; sdim++)
          q_ray_values_ptr[aii][sdim] =  -r_dir_ptr[rii][sdim];  // Ray dir goes to first 3 components.
        q_ray_values_ptr[aii][space_dim] = 0.0;                  // Ray doesn't contribute to field component.
          // Control point index.
        const int32 offset = 2*aii;
        q_ray_ctrl_idx_ptr[offset] = size_a_queries;    // From 0...
        q_ray_ctrl_idx_ptr[offset + 1] = aii;        // ...toward dir (and beyond).

        // Ray query parameters.
        q_ray_el_id_ptr[rii] = aii;               // Set active queries one-to-one with ray "field" elements.
        q_ray_ref_pt_ptr[rii][0] = r_dist_ptr[rii];  // Insert ray distance as initial guess (ray "reference" coordinate).

        // Target.
        for (int32 sdim = 0; sdim < space_dim; sdim++)
          target_ptr[rii][sdim] = r_orig_ptr[rii][sdim];   // Ray orig goes to first 3 components.
        target_ptr[rii][space_dim] = isoval;               // Isovalue goes into physical field component.
      }
    });
  } //scope

  // ---------------------------------
  // At this point we have initialized
  // - q_ray_ref.m_eltrans,
  // - q_ray_ref.m_el_ids, and
  // - q_ray_ref.m_ref_pts.
  //
  // We have also initialized
  // - q_meshfield_ref.m_el_ids,
  // - q_meshfield_ref.m_ref_pts, and
  // - target.
  //
  // We still need to initialize q_meshfield_ref.m_eltrans.
  // --------------------------------------------------

  //
  // Associate data from MeshField.
  //

  // Try to assert that the mesh elements and the field elements are 1-to-1.
  // For example, don't combine a nonconforming spatial field and a conforming scalar field.
  {
    assert(m_eltrans_space.get_el_dofs() == m_eltrans_field.get_el_dofs());
    assert(m_eltrans_space.get_size_el() == m_eltrans_field.get_size_el());
    assert(m_eltrans_space.get_size_ctrl() == m_eltrans_field.get_size_ctrl());
  }
    const int32 p_order = m_eltrans_space.get_m_shape().m_p_order;
    const int32 el_dofs = m_eltrans_space.get_el_dofs();
    const int32 mesh_field_size_dofs = m_size_el * el_dofs;
    const int32 size_ctrl = m_eltrans_space.get_size_ctrl();
  {
    const int32 *space_ctrl_idx_ptr = m_eltrans_space.get_m_ctrl_idx_const().get_device_ptr_const();
    const int32 *field_ctrl_idx_ptr = m_eltrans_field.get_m_ctrl_idx_const().get_device_ptr_const();
    RAJA::ReduceMin<reduce_policy, bool> all_equal(true);  // The collective MIN of Boolean values <-> collective AND.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, mesh_field_size_dofs), [=] DRAY_LAMBDA (int32 dof_idx)
    {
      // Compare ctrl_idx for corresponding dofs in spatial and scalar fields.
      // If they are not equal, then the result of the whole reduce will be 'false'.
      const bool current_equal = ( space_ctrl_idx_ptr[dof_idx] == field_ctrl_idx_ptr[dof_idx] );
      all_equal.min(current_equal);
    });
    assert(all_equal.get());
  }

  q_meshfield_ref.m_eltrans.resize(m_size_el, el_dofs, ETS::ShapeType::factory(p_order), size_ctrl);

  // Copy all ctrl_idx.
  {
    const int32 *space_ctrl_idx_ptr = m_eltrans_space.get_m_ctrl_idx_const().get_device_ptr_const();
    int32 *q_meshfield_ctrl_idx_ptr = q_meshfield_ref.m_eltrans.get_m_ctrl_idx().get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, mesh_field_size_dofs), [=] DRAY_LAMBDA (int32 dof_idx)
    {
      q_meshfield_ctrl_idx_ptr[dof_idx] = space_ctrl_idx_ptr[dof_idx];
    });
  }

  // Copy all control point values.
  {
    const Vec<T,space_dim> *space_values_ptr = m_eltrans_space.get_m_values_const().get_device_ptr_const();
    const Vec<T,field_dim> *field_values_ptr = m_eltrans_field.get_m_values_const().get_device_ptr_const();
    Vec<T, space_dim + field_dim> *meshfield_values_ptr = q_meshfield_ref.m_eltrans.get_m_values().get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_ctrl), [=] DRAY_LAMBDA (int32 ctrl_idx)
    {
      int32 dim = 0;
      for (int32 sdim = 0; sdim < space_dim; sdim++, dim++)
        meshfield_values_ptr[ctrl_idx][dim] = space_values_ptr[ctrl_idx][sdim];
      for (int32 fdim = 0; fdim < field_dim; fdim++, dim++)
        meshfield_values_ptr[ctrl_idx][dim] = field_values_ptr[ctrl_idx][fdim];
    });
  }

  //
  // NewtonSolve this system.
  //
  Array<int32> solve_status;
  int32 num_steps = NewtonSolve<QMeshFieldRay>::step(target, intersection_system, active_queries, solve_status, 10);

  // For debugging, set the ray hit_ref_pt to a well-defined dummy value.
  {
    const int32 *active_idx_ptr = rays.m_active_rays.get_device_ptr_const();
    Vec<T,space_dim> *r_hit_ref_pt_ptr = rays.m_hit_ref_pt.get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_a_rays), [=] DRAY_LAMBDA (int32 aii)
    {
      const int32 rii = active_idx_ptr[aii];
      const Vec<T,space_dim> black = {0, 0, 0};
      r_hit_ref_pt_ptr[rii] = black;
    });
  }
  
  // The effects of NewtonSolve::step() are saved in q_meshfield_ref.m_ref_pts (and the results fields).
  // Send the results back into the parameter "rays" (m_dist, m_hit_ref_pt, m_hit_idx).
  {
    const int32 *active_idx_ptr = active_queries.get_device_ptr_const();
    const int32 *solve_status_ptr = solve_status.get_device_ptr_const();
    const Vec<T,ref_dim> *q_meshfield_ref_pt_ptr = q_meshfield_ref.m_ref_pts.get_device_ptr_const();
    const Vec<T,1> *q_ray_ref_pt_ptr = q_ray_ref.m_ref_pts.get_device_ptr_const();
    Vec<T,ref_dim> *r_hit_ref_pt_ptr = rays.m_hit_ref_pt.get_device_ptr();
    T *r_dist_ptr = rays.m_dist.get_device_ptr();
    int32 *r_hit_idx_ptr = rays.m_hit_idx.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_a_queries), [=] DRAY_LAMBDA (int32 aii)
    //for (int32 aii = 0; aii < size_a_queries; aii++)
    {
      const int32 rii = active_idx_ptr[aii];

      for (int32 rdim = 0; rdim < ref_dim; rdim++)
        r_hit_ref_pt_ptr[rii][rdim] = q_meshfield_ref_pt_ptr[rii][rdim];
      r_dist_ptr[rii] = q_ray_ref_pt_ptr[rii][0];

      if (solve_status_ptr[aii] == NewtonSolve<QMeshFieldRay>::NotConverged
          || !ETS::ShapeType::IsInside(q_meshfield_ref_pt_ptr[rii]) )
      {
        r_hit_idx_ptr[rii] = -1;

        /// // For debugging, set the ray hit_ref_pt to a well-defined dummy value.
        /// if (solve_status_ptr[aii] == NewtonSolve<QMeshFieldRay>::NotConverged)
        ///   r_hit_ref_pt_ptr[rii] = { 1, 0, 0 };    // Dummy output (red).
        /// else if (!ETS::ShapeType::IsInside(q_meshfield_ref_pt_ptr[rii]))
        ///   r_hit_ref_pt_ptr[rii] = { 0, 0, 1 };    // Dummy output (blue).
      }
    });
  }

}


/*
 * Returns shading context of size rays.
 * This keeps image-bound buffers aligned with rays.
 * For inactive rays, the is_valid flag is set to false.
 */
template <typename T, class ETS, class ETF>
ShadingContext<T>
MeshField<T,ETS,ETF>::get_shading_context(Ray<T> &rays) const
{
  const int32 size_rays = rays.size();
  //const int32 size_active_rays = rays.m_active_rays.size();

  ///std::cout << "MeshField::get_shading_context() - rays.m_hit_idx gathered by m_active_rays: " << std::endl;
  ///gather(rays.m_hit_idx, rays.m_active_rays).summary();

  // Refine the set of active rays by culling any rays with m_hit_idx == -1.
  Array<int32> active_valid_idx = compact(rays.m_active_rays, rays.m_hit_idx, IsNonnegative<int32>());
  const int32 size_active_valid = active_valid_idx.size();

  ///std::cout << "MeshField::get_shading_context() - active_valid_idx == ";
  ///active_valid_idx.summary();

  ShadingContext<T> shading_ctx;
  shading_ctx.resize(size_rays);

  // Initialize outputs to well-defined dummy values (except m_pixel_id and m_ray_dir; see below).
  const Vec<T,3> one_two_three = {123., 123., 123.};
  array_memset_vec(shading_ctx.m_hit_pt, one_two_three);
  array_memset_vec(shading_ctx.m_normal, one_two_three);
  array_memset(shading_ctx.m_sample_val, static_cast<T>(-3.14));
  array_memset(shading_ctx.m_is_valid, static_cast<int32>(0));   // All are initialized to "invalid."
  
  // Adopt the fields (m_pixel_id) and (m_dir) from rays to intersection_ctx.
  shading_ctx.m_pixel_id = rays.m_pixel_id;
  shading_ctx.m_ray_dir = rays.m_dir;

  // TODO cache this in a field of MFEMGridFunction.
  T field_min, field_max;
  field_bounds(field_min, field_max);
  T field_range_rcp = rcp_safe(field_max - field_min);

  // Compute field value and derivative at reference points.
  typedef ElTransQuery<ETF> FQueryT;
  FQueryT field_query;
  field_query.m_eltrans = m_eltrans_field;
  field_query.m_el_ids = rays.m_hit_idx;
  field_query.m_ref_pts = rays.m_hit_ref_pt;
  field_query.m_result_val.resize(size_rays);
  field_query.m_result_deriv.resize(size_rays);
  field_query.query( active_valid_idx );

  // Compute space derivative (need inverse Jacobian to finish computing gradient).
  typedef ElTransQuery<ETS> SQueryT;
  SQueryT space_query;
  space_query.m_eltrans = m_eltrans_space;
  space_query.m_el_ids = rays.m_hit_idx;
  space_query.m_ref_pts = rays.m_hit_ref_pt;
  space_query.m_result_val.resize(size_rays);  // Unused but needed anyway.
  space_query.m_result_deriv.resize(size_rays);
  space_query.query( active_valid_idx );

  const int32 *hit_idx_ptr = rays.m_hit_idx.get_device_ptr_const();
  ///const Vec<T,3> *hit_ref_pt_ptr = rays.m_hit_ref_pt.get_device_ptr_const();

  int32 *is_valid_ptr = shading_ctx.m_is_valid.get_device_ptr();
  T *sample_val_ptr = shading_ctx.m_sample_val.get_device_ptr();
  Vec<T,3> *normal_ptr = shading_ctx.m_normal.get_device_ptr();
  //Vec<T,3> *hit_pt_ptr = shading_ctx.m_hit_pt.get_device_ptr();

  const int32 *active_valid_ptr = active_valid_idx.get_device_ptr_const();

  const typename FQueryT::ptr_bundle_const_t field_val_ptr = field_query.get_val_device_ptr_const();
  const typename FQueryT::ptr_bundle_const_t field_deriv_ptr = field_query.get_val_device_ptr_const();
  const typename SQueryT::ptr_bundle_const_t space_deriv_ptr = space_query.get_deriv_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active_valid), [=] DRAY_LAMBDA (int32 aray_idx)
  {
    const int32 ray_idx = active_valid_ptr[aray_idx];

    if (hit_idx_ptr[ray_idx] == -1)
    {
      // Sample is not in an element.
      is_valid_ptr[ray_idx] = 0;

      // This should never happen, now that we have created active_valid_idx.
    }
    else
    {
      // Sample is in an element.
      is_valid_ptr[ray_idx] = 1;

      // Get scalar field value and copy to output.
      const Vec<T,1> field_val = FQueryT::get_val(field_val_ptr, ray_idx);
      sample_val_ptr[ray_idx] = (field_val[0] - field_min) * field_range_rcp;

      // Get "gradient hat," derivative of scalar field.
      const Matrix<T,1,3> gradient_h = FQueryT::get_deriv(field_deriv_ptr, ray_idx);

      // Get spatial derivative and compute spatial field gradient as g = gh * J_inv.
      const Matrix<T,3,3> jacobian = SQueryT::get_deriv(space_deriv_ptr, ray_idx);
      bool inv_valid;
      const Matrix<T,3,3> j_inv = matrix_inverse(jacobian, inv_valid);
      //TODO How to handle the case that inv_valid == false?
      const Matrix<T,1,3> gradient_mat = gradient_h * j_inv;
      Vec<T,3> gradient = gradient_mat.get_row(0);
      
      // Normalize gradient vector and copy to output.
      T gradient_mag = gradient.magnitude();
      gradient.normalize();   //TODO What if the gradient is (0,0,0)?
      normal_ptr[ray_idx] = gradient;

      //TODO store the magnitude of the gradient if that is desired.

      //TODO compute hit point using ray origin, direction, and distance.
    }
  });

  return shading_ctx;
}


// Make "static constexpr" work with some linkers.
template <typename T, class ETS, class ETF> constexpr int32 MeshField<T,ETS,ETF>::ref_dim;
template <typename T, class ETS, class ETF> constexpr int32 MeshField<T,ETS,ETF>::space_dim;
template <typename T, class ETS, class ETF> constexpr int32 MeshField<T,ETS,ETF>::field_dim;

// Explicit instantiations.
template class MeshField<float32, ElTrans_BernsteinShape<float32, 3,3>, ElTrans_BernsteinShape<float32, 1,3>>;
template class MeshField<float64, ElTrans_BernsteinShape<float64, 3,3>, ElTrans_BernsteinShape<float64, 1,3>>;


} // namespace dray

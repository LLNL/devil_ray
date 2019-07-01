#include <dray/filters/internal/get_shading_context.hpp>

namespace dray
{
namespace internal
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

template
Array<ShadingContext<float32>>
get_shading_context<float32>(Array<Ray<float32>> &rays,
                             Field<float32> &field,
                             Mesh<float32> &mesh,
                             Array<RefPoint<float32,3>> &rpoints);

template
Array<ShadingContext<float64>>
get_shading_context<float64>(Array<Ray<float64>> &rays,
                             Field<float64> &field,
                             Mesh<float64> &mesh,
                             Array<RefPoint<float64,3>> &rpoints);

} // namespace internal
} // namespace dray

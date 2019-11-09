#include <dray/filters/internal/get_shading_context.hpp>
#include <dray/policies.hpp>

namespace dray
{
namespace internal
{

  // --------------------------------------------------------------------
  // MatInvHack: Don't call inverse() if we can't, but get past compiler.
  // --------------------------------------------------------------------
  template <typename T, int32 M, int32 N>
  struct MatInvHack
  {
    DRAY_EXEC static Matrix<T,N,M>
    get_inverse(const Matrix<T,M,N> &m, bool &valid)
    {
      Matrix<T,N,M> a;
      a.identity();
      valid = false;
      return a;
    }
  };
  // ----------------------------------------
  template <typename T, int32 S>
  struct MatInvHack<T, S, S>
  {
    DRAY_EXEC static Matrix<T,S,S>
    get_inverse(const Matrix<T,S,S> &m, bool &valid)
    {
      return matrix_inverse(m, valid);
    }
  };
  // ----------------------------------------


template <class ElemT>
Array<ShadingContext>
get_shading_context(Array<Ray> &rays,
                    Range<float32> scalar_range,
                    Field<FieldOn<ElemT, 1u>> &field,
                    Mesh<ElemT> &mesh,
                    Array<RefPoint<ElemT::get_dim()>> &rpoints)
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

  // Convention: If dim==2, use surface normal as direction.
  //             If dim==3, use field gradient as direction.
  //             In any case, make sure it faces the camera.

  constexpr int32 dim = ElemT::get_dim();

  const int32 size_rays = rays.size();
  //const int32 size_active_rays = rays.m_active_rays.size();

  Array<ShadingContext> shading_ctx;
  shading_ctx.resize(size_rays);
  ShadingContext *ctx_ptr = shading_ctx.get_device_ptr();

  // Adopt the fields (m_pixel_id) and (m_dir) from rays to intersection_ctx.
  //shading_ctx.m_pixel_id = rays.m_pixel_id;
  //shading_ctx.m_ray_dir = rays.m_dir;

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<Float,3> one_two_three = {123., 123., 123.};

  const int32 size = rays.size();

  const Float field_min = scalar_range.min();
  const Float field_range_rcp = rcp_safe( scalar_range.length() );

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const RefPoint<dim> *rpoints_ptr = rpoints.get_device_ptr_const();

  MeshAccess<ElemT> device_mesh = mesh.access_device_mesh();
  FieldAccess<FieldOn<ElemT, 1u>> device_field = field.access_device_field();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    ShadingContext ctx;
    // TODO: create struct initializers
    ctx.m_hit_pt = one_two_three;
    ctx.m_normal = one_two_three;
    ctx.m_sample_val = 3.14f;
    ctx.m_gradient_mag = 55.55f;

    const Ray &ray = ray_ptr[i];
    const RefPoint<dim> &rpt = rpoints_ptr[i];

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

      const int32 el_id = rpt.m_el_id;
      const Vec<Float, dim> ref_pt = rpt.m_el_coords;

      // Evaluate element transformation and scalar field.
      Vec<Vec<Float, 3>, dim> jac_vec;
      Vec<Float, 3> world_pos = device_mesh.get_elem(el_id).eval_d(ref_pt, jac_vec);

      Vec<Float, 1> field_val;
      Vec<Vec<Float, 1>, dim> field_deriv;  // Only init'd if dim==3.
      if (dim == 2)
        field_val = device_field.get_elem(el_id).eval(ref_pt);
      else if (dim == 3)
        field_val = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv);

      // What we output as the normal depends if dim==2 or 3.
      if (dim == 2)
      {
        // Use the normalized cross product of the jacobian
        ctx.m_normal = cross(jac_vec[0], jac_vec[1]);
      }
      else if (dim == 3)
      {
        // Use the gradient of the scalar field relative to world axes.
        Matrix<Float, 3, dim> jacobian_matrix;
        Matrix<Float, 1, dim> gradient_ref;
        for (int32 rdim = 0; rdim < 3; rdim++)
        {
          jacobian_matrix.set_col(rdim, jac_vec[rdim]);
          gradient_ref.set_col(rdim, field_deriv[rdim]);
        }

        // To convert to world coords, use g = gh * J_inv.
        bool inv_valid;
        const Matrix<Float, dim, 3> j_inv =
            MatInvHack<Float, 3, dim>::get_inverse(jacobian_matrix, inv_valid);
        //TODO How to handle the case that inv_valid == false?
        const Matrix<Float, 1, 3> gradient_mat = gradient_ref * j_inv;
        Vec<Float,3> gradient_world = gradient_mat.get_row(0);

        // Output.
        ctx.m_normal = gradient_world;
        ctx.m_gradient_mag = gradient_world.magnitude();
        //TODO What if the gradient is (0,0,0)?
      }

      // Finalize the normal.
      ctx.m_normal.normalize();
      if (dot(ctx.m_normal, ray.m_dir) > 0.0f)
        ctx.m_normal = -ctx.m_normal;

      // Output scalar field value.
        // TODO: deffer this calculation into the shader
        // output actual scalar value and normal
      ctx.m_sample_val = (field_val[0] - field_min) * field_range_rcp;
    }

    ctx_ptr[i] = ctx;

  });

  return shading_ctx;
}

template <class ElemT>
Array<Fragment>
get_fragments(Array<Ray> &rays,
              Range<float32> scalar_range,
              Field<FieldOn<ElemT, 1u>> &field,
              Mesh<ElemT> &mesh,
              Array<RayHit>> &hits)
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

  // Convention: If dim==2, use surface normal as direction.
  //             If dim==3, use field gradient as direction.
  //             In any case, make sure it faces the camera.

  constexpr int32 dim = ElemT::get_dim();

  const int32 size_rays = rays.size();
  //const int32 size_active_rays = rays.m_active_rays.size();

  Array<Fragment> fragments;
  fragments.resize(size_rays);
  Fragment *fragments_ptr = fragments.get_device_ptr();

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<Float,3> one_two_three = {123., 123., 123.};

  const int32 size = rays.size();

  const Float field_min = scalar_range.min();
  const Float field_range_rcp = rcp_safe( scalar_range.length() );

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const RayHit *hit_ptr = hits.get_device_ptr_const();

  MeshAccess<ElemT> device_mesh = mesh.access_device_mesh();
  FieldAccess<FieldOn<ElemT, 1u>> device_field = field.access_device_field();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Fragment frag;
    frag.m_scalar = -1.f;
    frag.m_normal = {-1.f, -1.f, -1.f};

    const Ray &ray = ray_ptr[i];
    const Hit &hit = hit_ptr[i];

    if(hit.m_hit_idx > -1)
    {
      const int32 el_id = rpt.m_el_id;
      const Vec<Float, dim> ref_pt;
      ref_pt[0] = hit.m_ref_pt[0];
      ref_pt[1] = hit.m_ref_pt[1];
      if(dim == 3)
      {
        ref_pt[2] = hit.m_ref_pt[2];
      }
      // Evaluate element transformation and scalar field.
      Vec<Vec<Float, 3>, dim> jac_vec;
      Vec<Float, 3> world_pos = device_mesh.get_elem(el_id).eval_d(ref_pt, jac_vec);

      Vec<Float, 1> field_val;
      Vec<Vec<Float, 1>, dim> field_deriv;  // Only init'd if dim==3.

      if (dim == 2)
        frag.m_scalar = device_field.get_elem(el_id).eval(ref_pt);
      else if (dim == 3)
        frag.m_scalar = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv);

      // What we output as the normal depends if dim==2 or 3.
      if (dim == 2)
      {
        // Use the normalized cross product of the jacobian
        frag.m_normal = cross(jac_vec[0], jac_vec[1]);
      }
      else if (dim == 3)
      {
        // Use the gradient of the scalar field relative to world axes.
        Matrix<Float, 3, dim> jacobian_matrix;
        Matrix<Float, 1, dim> gradient_ref;
        for (int32 rdim = 0; rdim < 3; rdim++)
        {
          jacobian_matrix.set_col(rdim, jac_vec[rdim]);
          gradient_ref.set_col(rdim, field_deriv[rdim]);
        }

        // To convert to world coords, use g = gh * J_inv.
        bool inv_valid;
        const Matrix<Float, dim, 3> j_inv =
            MatInvHack<Float, 3, dim>::get_inverse(jacobian_matrix, inv_valid);
        //TODO How to handle the case that inv_valid == false?
        const Matrix<Float, 1, 3> gradient_mat = gradient_ref * j_inv;
        Vec<Float,3> gradient_world = gradient_mat.get_row(0);

        // Output.
        hit.m_normal = gradient_world;
        //TODO What if the gradient is (0,0,0)? (Matt: it will be bad)
      }
    }

    fragments_ptr[i] = frag;

  });

  return shading_ctx;
}

/// template <typename T, class ElemT>
/// Array<ShadingContext<T>>
/// get_shading_context(Array<Ray<T>> &rays,
///                     Array<Vec<int32,2>> &faces,
///                     Field<T, FieldOn<ElemT, 1u>> &field,
///                     Mesh<T, ElemT> &mesh,
///                     Array<RefPoint<T,3>> &rpoints)
/// {
///   const int32 size_rays = rays.size();
///   //const int32 size_active_rays = rays.m_active_rays.size();
///
///   Array<ShadingContext<T>> shading_ctx;
///   shading_ctx.resize(size_rays);
///   ShadingContext<T> *ctx_ptr = shading_ctx.get_device_ptr();
///
///   // Initialize other outputs to well-defined dummy values.
///   constexpr Vec<T,3> one_two_three = {123., 123., 123.};
///
///   const int32 size = rays.size();
///
///   const Range<> field_range = field.get_range();
///   const T field_min = field_range.min();
///   const T field_range_rcp = rcp_safe( field_range.length() );
///
///   const Ray<T> *ray_ptr = rays.get_device_ptr_const();
///   const RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr_const();
///
///   MeshAccess<T, ElemT> device_mesh = mesh.access_device_mesh();
///   FieldAccess<T, FieldOn<ElemT, 1u>> device_field = field.access_device_field();
///   const Vec<int32,2> *faces_ptr = faces.get_device_ptr_const();
///
///   RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
///   {
///
///     ShadingContext<T> ctx;
///     // TODO: create struct initializers
///     ctx.m_hit_pt = one_two_three;
///     ctx.m_normal = one_two_three;
///     ctx.m_sample_val = 3.14f;
///     ctx.m_gradient_mag = 55.55f;
///
///     const Ray<T> &ray = ray_ptr[i];
///     const RefPoint<T,3> &rpt = rpoints_ptr[i];
///
///     ctx.m_pixel_id = ray.m_pixel_id;
///     ctx.m_ray_dir  = ray.m_dir;
///
///     if (rpt.m_el_id == -1)
///     {
///       // There is no intersection.
///       ctx.m_is_valid = 0;
///     }
///     else
///     {
///       // There is an intersection.
///       ctx.m_is_valid = 1;
///     }
///
///     if(ctx.m_is_valid)
///     {
///       // Compute hit point using ray origin, direction, and distance.
///       ctx.m_hit_pt = ray.m_orig + ray.m_dir * ray.m_dist;
///
///       // Evaluate element transformation to get scalar field value and gradient.
///
///       const Vec<int32,2> face_id = faces_ptr[rpt.m_el_id];
///       const int32 el_id = face_id[0];
///       const Vec<T,3> ref_pt = rpt.m_el_coords;
///
///       Vec<T,3> world_pos, world_normal;
///
///       const FaceElement<T,3> face_el
///         = device_mesh.get_elem(el_id).get_face_element(face_id[1]);
///
///       face_normal_and_position(face_el,
///                                ref_pt,
///                                world_pos,
///                                world_normal);
///
///       ctx.m_hit_pt = world_pos;
///
///       if (dot(world_normal, ray.m_dir) > 0.0f)
///       {
///         world_normal = -world_normal;   //Flip back toward camera.
///       }
///
///       ctx.m_normal = world_normal;
///
///       Vec<T,1> field_val;
///       Vec<Vec<T,1>,3> field_deriv;
///       field_val = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv);
///
///       // TODO: deffer this calculation into the shader
///       // output actual scalar value and normal
///       ctx.m_sample_val = (field_val[0] - field_min) * field_range_rcp;
///     }
///
///     ctx_ptr[i] = ctx;
///
///   });
///
///   return shading_ctx;
/// }

// <2D>
template
Array<ShadingContext>
get_shading_context<>(Array<Ray> &rays,
                      Range<Float> scalar_range,
                      Field<Element<2u, 1u, ElemType::Quad, Order::General>> &field,
                      Mesh<MeshElem<2u, ElemType::Quad, Order::General>> &mesh,
                      Array<RefPoint<2>> &rpoints);

// <3D>
template
Array<ShadingContext>
get_shading_context<>(Array<Ray> &rays,
                      Range<Float> scalar_range,
                      Field<Element<3u, 1u, ElemType::Quad, Order::General>> &field,
                      Mesh<MeshElem<3u, ElemType::Quad, Order::General>> &mesh,
                      Array<RefPoint<3>> &rpoints);
} // namespace internal
} // namespace dray

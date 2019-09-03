#include <dray/utils/ray_utils.hpp>
#include <dray/policies.hpp>
#include <dray/transform_3d.hpp>

#include <assert.h>

namespace dray
{
namespace detail
{
template<typename T>
Array<float32>
get_gl_depth_buffer(const Array<Ray<T>> &rays,
                    const Camera &camera,
                    const float32 near,
                    const float32 far)
{
  int32 size = rays.size();
  int32 image_size = camera.get_width() * camera.get_height();

  const Ray<T> *ray_ptr = rays.get_device_ptr_const();

  Array<float32> dbuffer;
  dbuffer.resize(image_size);
  array_memset(dbuffer, 1.0001f);

  float32 *d_ptr = dbuffer.get_device_ptr();
  Matrix<float32,4,4> view_proj = camera.projection_matrix(near, far) * camera.view_matrix();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 offset = ray_ptr[i].m_pixel_id;

    if(ray_ptr[i].m_near < ray_ptr[i].m_far && ray_ptr[i].m_dist < ray_ptr[i].m_far)
    {
      const Ray<T> ray = ray_ptr[i];
      const Vec<T,3> hit = ray.m_orig + ray.m_dir * ray.m_dist;
      Vec<float32,3> hitf;
      hitf[0] = static_cast<float32>(hit[0]);
      hitf[1] = static_cast<float32>(hit[1]);
      hitf[2] = static_cast<float32>(hit[2]);
      Vec<float32,3> transformed = transform_point(view_proj, hitf);
      float32 depth = 0.5f * transformed[2] + 0.5f;
      d_ptr[ray.m_pixel_id] = depth;
    }
  });

  return dbuffer;
}

} // namespace detail

Array<float32>
get_gl_depth_buffer(const Array<Ray<float32>> &rays,
                    const Camera &camera,
                    const float32 near,
                    const float32 far)
{
  return detail::get_gl_depth_buffer(rays,camera, near, far);
}

Array<float32>
get_gl_depth_buffer(const Array<Ray<float64>> &rays,
                    const Camera &camera,
                    const float32 near,
                    const float32 far)
{
  return detail::get_gl_depth_buffer(rays,camera, near, far);
}

} // namespace dray

#include <dray/utils/ray_utils.hpp>
#include <dray/policies.hpp>

#include <assert.h>

namespace dray
{
namespace detail
{
template<typename T>
Array<float32>
get_depth_buffer(const Array<Ray<T>> &rays,
                 const int width,
                 const int height)
{
  int32 size = rays.size();
  int32 image_size = width * height;

  const Ray<T> *ray_ptr = rays.get_device_ptr_const();

  Array<float32> dbuffer;
  dbuffer.resize(image_size);
  array_memset(dbuffer, infinity<float32>());

  float32 *d_ptr = dbuffer.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 offset = ray_ptr[i].m_pixel_id;

    if(ray_ptr[i].m_near < ray_ptr[i].m_far && ray_ptr[i].m_dist < ray_ptr[i].m_far)
    {
      d_ptr[i] = ray_ptr[i].m_dist;
    }
  });

  return dbuffer;
}

} // namespace detail

Array<float32>
get_depth_buffer(const Array<Ray<float32>> &rays,
                 const int width,
                 const int height)
{
  return detail::get_depth_buffer(rays, width, height);
}

Array<float32>
get_depth_buffer(const Array<Ray<float64>> &rays,
                 const int width,
                 const int height)
{
  return detail::get_depth_buffer(rays, width, height);
}

} // namespace dray

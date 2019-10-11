#include <dray/utils/ray_utils.hpp>
#include <dray/policies.hpp>
#include <dray/transform_3d.hpp>

#include <assert.h>

namespace dray
{
namespace detail
{
Array<float32>
get_gl_depth_buffer(const Array<Ray> &rays,
                    const Camera &camera,
                    const float32 near,
                    const float32 far)
{
  int32 size = rays.size();
  int32 image_size = camera.get_width() * camera.get_height();

  const Ray *ray_ptr = rays.get_device_ptr_const();

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
      const Ray ray = ray_ptr[i];
      const Vec<Float,3> hit = ray.m_orig + ray.m_dir * ray.m_dist;
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
get_gl_depth_buffer(const Array<Ray> &rays,
                    const Camera &camera,
                    const float32 near,
                    const float32 far)
{
  return detail::get_gl_depth_buffer(rays,camera, near, far);
}

Array<float32>
get_depth_buffer_img(const Array<Ray> &rays,
                     const int width,
                     const int height)
{
  Float minv = 1000000.f;
  Float maxv = -1000000.f;

  int32 size = rays.size();
  int32 image_size = width * height;

  const Ray *ray_ptr = rays.get_host_ptr_const();

  for(int32 i = 0; i < size;++i)
  {
    if(ray_ptr[i].m_near < ray_ptr[i].m_far && ray_ptr[i].m_dist < ray_ptr[i].m_far)
    {
      Float depth = ray_ptr[i].m_dist;
      minv = fminf(minv, depth);
      maxv = fmaxf(maxv, depth);
    }
  }

  Array<float32> dbuffer;
  dbuffer.resize(image_size* 4);
  array_memset_zero(dbuffer);

  float32 *d_ptr = dbuffer.get_host_ptr();
  for (int32 i = 0; i < image_size; i++)
  {
    d_ptr[i + 0] = 0.0f;
    d_ptr[i + 1] = 0.0f;
    d_ptr[i + 2] = 0.0f;
    d_ptr[i + 3] = 1.0f;
  }


  float32 len = maxv - minv;

  for(int32 i = 0; i < size;++i)
  {
    int32 offset = ray_ptr[i].m_pixel_id * 4;
    float32 val = 0;
    if(ray_ptr[i].m_near < ray_ptr[i].m_far && ray_ptr[i].m_dist < ray_ptr[i].m_far)
    {
      val = (ray_ptr[i].m_dist - minv) / len;
    }
    d_ptr[offset + 0] = val;
    d_ptr[offset + 1] = val;
    d_ptr[offset + 2] = val;
    d_ptr[offset + 3] = 1.f;
  }

  return dbuffer;
}

void save_depth(const Array<Ray> &rays,
                const int width,
                const int height,
                std::string file_name)
{

  Array<float32> dbuffer = get_depth_buffer_img(rays, width, height);
  float32 *d_ptr = dbuffer.get_host_ptr();

  PNGEncoder encoder;
  encoder.encode(d_ptr, width, height);
  encoder.save(file_name + ".png");
}

} // namespace dray

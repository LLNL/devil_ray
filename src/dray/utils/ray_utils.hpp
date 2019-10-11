#ifndef DRAY_RAY_UTILS_HPP
#define DRAY_RAY_UTILS_HPP

#include <dray/utils/png_encoder.hpp>
#include <dray/camera.hpp>
#include <dray/types.hpp>
#include <dray/ray.hpp>

#include <dray/array_utils.hpp>
#include <dray/vec.hpp>   //don't really need.
#include <dray/aabb.hpp>

#include <string>

namespace dray
{

DRAY_EXEC
bool intersect_ray_aabb(const Ray &ray, const AABB<3> &aabb)
{
  const Vec<Float,3> dir_rcp = {rcp_safe(ray.m_dir[0]), rcp_safe(ray.m_dir[1]), rcp_safe(ray.m_dir[2])};
  const Range<> (&aabbr)[3] = aabb.m_ranges;

  const Float xmin = (aabbr[0].min() - ray.m_orig[0]) * dir_rcp[0];
  const Float ymin = (aabbr[1].min() - ray.m_orig[1]) * dir_rcp[1];
  const Float zmin = (aabbr[2].min() - ray.m_orig[2]) * dir_rcp[2];
  const Float xmax = (aabbr[0].max() - ray.m_orig[0]) * dir_rcp[0];
  const Float ymax = (aabbr[1].max() - ray.m_orig[1]) * dir_rcp[1];
  const Float zmax = (aabbr[2].max() - ray.m_orig[2]) * dir_rcp[2];

  Float left  = fmaxf(fmaxf( fminf(xmin,xmax), fminf(ymin,ymax)), fminf(zmin,zmax));
  Float right = fminf(fminf( fmaxf(xmin,xmax), fmaxf(ymin,ymax)), fmaxf(zmin,zmax));

  return left <= right;
}

// a single channel image of the depth buffer
Array<float32>
get_gl_depth_buffer(const Array<Ray> &rays,
                    const Camera &camera,
                    const float32 near,
                    const float32 far);

// a grey scale image of the depth buffer
Array<float32>
get_depth_buffer_img(const Array<Ray> &rays,
                     const int width,
                     const int height);

void save_depth(const Array<Ray> &rays,
                const int width,
                const int height,
                std::string file_name = "depth");

/**
 * This function assumes that rays are grouped into bundles each of size (num_samples),
 * and each bundle belongs to the same pixel_id. For each pixel, we count the number of rays
 * which have hit something, and divide the total by (num_samples).
 * The result is a scalar value for each pixel. We output this result to a png image.
 */
//void save_hitrate(const Array<Ray> &rays, const int32 num_samples, const int width, const int height)
//{
//
//  int32 size = rays.size();
//  int32 image_size = width * height;
//
//  // Read-only host pointers to input ray fields.
//  const int32 *hit_ptr = rays.m_hit_idx.get_host_ptr_const();
//  const int32 *pid_ptr = rays.m_pixel_id.get_host_ptr_const();
//
//  // Result array where we store the normalized count of # hits per bundle.
//  // Values should be between 0 and 1.
//  Array<float32> img_buffer;
//  img_buffer.resize(image_size* 4);
//  float32 *img_ptr = img_buffer.get_host_ptr();
//
//  for (int32 px_channel_idx = 0; px_channel_idx < img_buffer.size(); px_channel_idx++)
//  {
//    //img_ptr[px_channel_idx] = 0;
//    img_ptr[px_channel_idx] = 1;
//  }
//
//  ///RAJA::forall<for_policy>(RAJA::RangeSegment(0, size / num_samples), [=] DRAY_LAMBDA (int32 bundle_idx)
//  for (int32 bundle_idx = 0; bundle_idx < size / num_samples; bundle_idx++)
//  {
//    const int32 b_offset = bundle_idx * num_samples;
//
//    int32 num_hits = 0;
//    for (int32 sample_idx = 0; sample_idx < num_samples; ++sample_idx)
//    {
//      num_hits += (hit_ptr[b_offset + sample_idx] != -1);
//    }
//
//    const float32 hitrate = num_hits / (float32) num_samples;
//
//    const int32 pixel_offset = pid_ptr[b_offset] * 4;
//    img_ptr[pixel_offset + 0] = 1.f - hitrate;
//    img_ptr[pixel_offset + 1] = 1.f - hitrate;
//    img_ptr[pixel_offset + 2] = 1.f - hitrate;
//    img_ptr[pixel_offset + 3] = 1.f;
//  }
//  ///});
//
//  PNGEncoder encoder;
//  encoder.encode(img_ptr, width, height);
//  encoder.save("hitrate.png");
//}

}
#endif



#ifndef DRAY_RAY_UTILS_HPP
#define DRAY_RAY_UTILS_HPP

#include <dray/utils/png_encoder.hpp>
#include <dray/types.hpp>
#include <dray/ray.hpp>

#include <dray/vec.hpp>   //don't really need.

#include <string>

namespace dray
{

template<typename T>
void save_depth(const Array<Ray<T>> &rays, const int width, const int height)
{

  T minv = 1000000.f;
  T maxv = -1000000.f;

  int32 size = rays.size();
  int32 image_size = width * height;

  const Ray<T> *ray_ptr = rays.get_host_ptr_const();

  for(int32 i = 0; i < size;++i)
  {
    if(ray_ptr[i].m_hit_idx != -1)
    {
      T depth = ray_ptr[i].m_dist;
      minv = fminf(minv, depth);
      maxv = fmaxf(maxv, depth);
    }
  }

  Array<float32> dbuffer;
  dbuffer.resize(image_size* 4);
  float32 *d_ptr = dbuffer.get_host_ptr();
  float32 len = maxv - minv;

  for(int32 i = 0; i < size;++i)
  {
    int32 offset = i * 4;
    float32 val = 0;
    if(ray_ptr[i].m_hit_idx != -1)
    {
      val = (ray_ptr[i].m_dist - minv) / len;
    }

    d_ptr[offset + 0] = val;
    d_ptr[offset + 1] = val;
    d_ptr[offset + 2] = val;
    d_ptr[offset + 3] = 1.f;
  }

  PNGEncoder encoder;
  encoder.encode(d_ptr, width, height);
  encoder.save("depth.png");
}


#ifdef DRAY_STATS
/**
 * save_wasted_steps()
 *
 * Visualize "wasted" Newton iterations per ray.
 */
template <typename T>
void save_wasted_steps(const Array<Ray<T>> &rays, const int width, const int height, std::string image_name)
{
  int32 size = rays.size();
  int32 image_size = width * height;

  // Read-only host pointers to input ray fields.
  const Ray<T> *ray_ptr = rays.get_host_ptr_const();

  // Get maximum of total steps per ray, for scaling.
  T maxv = -1000000;
  for(int32 i = 0; i < size;++i)
  {
    maxv = fmaxf(maxv, ray_ptr[i].m_total_steps);
  }

  Array<float32> img_buffer;
  img_buffer.resize(image_size* 4);
  float32 *img_ptr = img_buffer.get_host_ptr();

  // Init background to black, alpha=1.
  for (int32 px_idx = 0; px_idx < size; px_idx++)
  {
    T intensity_steps = ray_ptr[px_idx].m_total_steps / (.00001f + maxv);
    T inefficiency = ray_ptr[px_idx].m_wasted_steps / (.00001f + ray_ptr[px_idx].m_total_steps);

    T r = fminf(1.0, 2*inefficiency);
    T b = 1.0 - fmaxf(0.0, 2*inefficiency - 1.0);

    img_ptr[4*px_idx + 0] = r * intensity_steps;
    img_ptr[4*px_idx + 1] = 0;
    img_ptr[4*px_idx + 2] = b * intensity_steps;
    img_ptr[4*px_idx + 3] = 1;
  }

  PNGEncoder encoder;
  encoder.encode(img_ptr, width, height);
  encoder.save(image_name);
}
#endif

/**
 * This function assumes that rays are grouped into bundles each of size (num_samples),
 * and each bundle belongs to the same pixel_id. For each pixel, we count the number of rays
 * which have hit something, and divide the total by (num_samples).
 * The result is a scalar value for each pixel. We output this result to a png image.
 */
template<typename T>
void save_hitrate(const Ray<T> &rays, const int32 num_samples, const int width, const int height)
{

  int32 size = rays.size();
  int32 image_size = width * height;

  // Read-only host pointers to input ray fields.
  const int32 *hit_ptr = rays.m_hit_idx.get_host_ptr_const();
  const int32 *pid_ptr = rays.m_pixel_id.get_host_ptr_const();

  // Result array where we store the normalized count of # hits per bundle.
  // Values should be between 0 and 1.
  Array<float32> img_buffer;
  img_buffer.resize(image_size* 4);
  float32 *img_ptr = img_buffer.get_host_ptr();

  for (int32 px_channel_idx = 0; px_channel_idx < img_buffer.size(); px_channel_idx++)
  {
    //img_ptr[px_channel_idx] = 0;
    img_ptr[px_channel_idx] = 1;
  }

  ///RAJA::forall<for_policy>(RAJA::RangeSegment(0, size / num_samples), [=] DRAY_LAMBDA (int32 bundle_idx)
  for (int32 bundle_idx = 0; bundle_idx < size / num_samples; bundle_idx++)
  {
    const int32 b_offset = bundle_idx * num_samples;

    int32 num_hits = 0;
    for (int32 sample_idx = 0; sample_idx < num_samples; ++sample_idx)
    {
      num_hits += (hit_ptr[b_offset + sample_idx] != -1);
    }

    const float32 hitrate = num_hits / (float32) num_samples;

    const int32 pixel_offset = pid_ptr[b_offset] * 4;
    img_ptr[pixel_offset + 0] = 1.f - hitrate;
    img_ptr[pixel_offset + 1] = 1.f - hitrate;
    img_ptr[pixel_offset + 2] = 1.f - hitrate;
    img_ptr[pixel_offset + 3] = 1.f;
  }
  ///});

  PNGEncoder encoder;
  encoder.encode(img_ptr, width, height);
  encoder.save("hitrate.png");
}

}
#endif



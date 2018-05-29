#ifndef DRAY_RAY_UTILS_HPP
#define DRAY_RAY_UTILS_HPP

#include <dray/utils/png_encoder.hpp>
#include <dray/types.hpp>
#include <dray/ray.hpp>

namespace dray
{

template<typename T>
void save_depth(const Ray<T> &rays, const int width, const int height)
{

  T minv = 1000000.f;
  T maxv = -1000000.f;

  int32 size = rays.size();
  int32 image_size = width * height;

  const T *dst_ptr = rays.m_dist.get_host_ptr_const();
  const int32 *hit_ptr = rays.m_hit_idx.get_host_ptr_const();

  for(int32 i = 0; i < size;++i)
  {
    if(hit_ptr[i] != -1) 
    {
      T depth = dst_ptr[i]; 
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
    if(hit_ptr[i] != -1) 
    {
      val = (dst_ptr[i] - minv) / len;
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

}
#endif



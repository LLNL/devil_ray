#include "gtest/gtest.h"
#include <dray/ambient_occlusion.hpp>

TEST(dray_ambient_occlusion, dray_ambient_occlusion_basic)
{
  dray::Array<dray::Vec<dray::float32,3>> dummy_array_vec;
  dray::Array<dray::float32>              dummy_grayscale;
  dray::Vec<dray::float32,3>              dummy_vec;
  dray::Ray<dray::float32>                dummy_rays;
  dray::IntersectionContext<dray::float32>  dummy_intersection_ctx;

  //TODO properly ininitialize the arrays/vectors so they have enough room.
  
  //dray::AmbientOcclusion<dray::float32>::calc_occlusion(dummy_array_vec, dummy_array_vec, 10, dummy_grayscale);
  //dray::AmbientOcclusion<dray::float32>::calc_occlusion(dummy_rays, 10, dummy_grayscale);
  ///dray::AmbientOcclusion<dray::float32>::gen_occlusion(dummy_vec, dummy_vec, 10, dummy_rays);

  dummy_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(dummy_intersection_ctx, 10, 0.1f, 10.f);
}

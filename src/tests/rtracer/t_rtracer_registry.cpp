#include "gtest/gtest.h"
#include <rtracer/array.hpp>
#include <rtracer/array_registry.hpp>

TEST(rtracer_array, rtracer_registry_basic)
{
  rtracer::Array<int> int_array;
  int_array.resize(2);
  int *host = int_array.get_host_ptr();
  host[0] = 0; 
  host[1] = 1; 
  

  size_t dev_usage = rtracer::ArrayRegistry::device_usage();

  // we should not have allocated anything yet
  ASSERT_EQ(dev_usage, 0);
  int *dev = int_array.get_device_ptr();
  
  dev_usage = rtracer::ArrayRegistry::device_usage();
  // not we shold have two ints
  ASSERT_EQ(dev_usage, 2 * sizeof(int));
 
  rtracer::ArrayRegistry::release_device_res();
  dev_usage = rtracer::ArrayRegistry::device_usage();
  ASSERT_EQ(dev_usage, 0);
  
}

#include "gtest/gtest.h"
#include <rtracer/rtracer.hpp>
#include <rtracer/test.hpp>

TEST(rtracer_test, rtracer_test)
{
  rtracer::rtracer tracer;
  tracer.about();
  rtracer::Tester tester; 
  tester.raja_loop();
}

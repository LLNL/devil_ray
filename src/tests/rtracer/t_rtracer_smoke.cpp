#include "gtest/gtest.h"
#include <rtracer/rtracer.hpp>

TEST(rtracer_smoke, rtracer_about)
{
  rtracer::rtracer tracer;
  tracer.about();
}

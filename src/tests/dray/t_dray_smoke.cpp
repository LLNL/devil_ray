#include "gtest/gtest.h"
#include <dray/dray.hpp>

TEST(dray_smoke, dray_about)
{
  dray::dray tracer;
  tracer.about();
}

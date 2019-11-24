#include "gtest/gtest.h"
#include <dray/dray.hpp>
#include <dray/test.hpp>

TEST (dray_test, dray_test)
{
  dray::dray tracer;
  tracer.about ();
  dray::Tester tester;
  tester.raja_loop ();
}

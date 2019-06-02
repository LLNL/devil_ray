#include "gtest/gtest.h"
#include "test_config.h"

#include <dray/utils/appstats.hpp>
#include <dray/utils/global_share.hpp>

TEST(dray_stats, dray_stats_smoke)
{

  std::shared_ptr<dray::stats::AppStats> app_stats_ptr = dray::stats::global_app_stats.get_shared_ptr();

  if (app_stats_ptr->is_enabled())
  {
    printf("App stats is enabled.\n");
  }
  else
  {
    printf("App stats is disabled!\n");
  }
}

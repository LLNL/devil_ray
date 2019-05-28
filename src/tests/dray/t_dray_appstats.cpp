#include "gtest/gtest.h"
#include "test_config.h"

#include <dray/utils/appstats.hpp>
#include <dray/utils/global_share.hpp>

TEST(dray_mfem_blueprint, dray_mfem_blueprint)  //TODO change mfem_blueprint to something that makes sense?
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

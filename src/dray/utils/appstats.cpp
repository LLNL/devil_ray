#include <dray/utils/global_share.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/array.hpp>

namespace dray
{
  std::ostream& operator<<(std::ostream &os, const _AppStatsStruct &stats_struct)
  {
    os << "AppStatsStruct {??}";
    return os;
  }


  GlobalShare<AppStats> global_app_stats;
}

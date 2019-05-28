#include <dray/utils/global_share.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/array.hpp>

namespace dray
{
  std::ostream& operator<<(std::ostream &os, const _AppStatsStruct &stats_struct)
  {
    os << "t:" << stats_struct.m_total_tests
       << " h:" << stats_struct.m_total_hits
       << " t_iter:" << stats_struct.m_total_test_iterations
       << " h_iter:" << stats_struct.m_total_hit_iterations;
    return os;
  }


  GlobalShare<AppStats> global_app_stats;
}

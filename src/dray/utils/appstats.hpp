#ifndef DRAY_APPSTATS_HPP
#define DRAY_APPSTATS_HPP

#include <dray/utils/global_share.hpp>

#include <dray/array.hpp>
#include <dray/ray.hpp>
#include <dray/types.hpp>
#include <dray/exports.hpp>

#include <iostream>
#include <vector>
#include <utility>

namespace dray
{
namespace stats
{

struct MattStats
{
  int32 m_newton_iters; // total newton iterations
  int32 m_candidates;   // number of candidates testes
  int32 m_found;        // found (1) or not (0)

  void DRAY_EXEC construct()
  {
    m_newton_iters = 0;
    m_candidates = 0;
    m_found = 0;
  }
  friend std::ostream& operator<<(std::ostream &os, const MattStats &stats_struct);
};

class StatStore
{
protected:
  static std::vector<std::vector<std::pair<int32,MattStats>>> m_ray_stats;
  static std::vector<std::vector<std::pair<Vec<float32,3>,MattStats>>> m_point_stats;
public:
  static void add_ray_stats(Array<Ray<float32>> &rays, Array<MattStats> &stats);
  static void add_ray_stats(Array<Ray<float64>> &rays, Array<MattStats> &stats);

  static void add_point_stats(Array<Vec<float32,3>> &points, Array<MattStats> &stats);
  static void add_point_stats(Array<Vec<float64,3>> &points, Array<MattStats> &stats);

  static void write_ray_stats(const int32 width,const int32 height);
  static void write_point_stats(const std::string name);
  static void clear();

};

// AppStatStruct: Summary counters to track things like number of iterative solves per ray.
//
// Can replace with multiple structs and multple arrays later, if needed.
//
// Note: All attributes should have a fixed size (known at compile time).
//
struct _AppStatsStruct   // Can be applied to a ray or an element.
{
  int32 m_total_tests;   // candidates-per-ray or rays-per-element
  int32 m_total_hits;
  int32 m_total_test_iterations;
  int32 m_total_hit_iterations;

  void DRAY_EXEC construct()
  {
    m_total_tests = 0;
    m_total_hits = 0;
    m_total_test_iterations = 0;
    m_total_hit_iterations = 0;
  }

  friend std::ostream& operator<<(std::ostream &os, const _AppStatsStruct &stats_struct);
};


struct _AppStatsAccess
{
  _AppStatsStruct *m_query_stats_ptr;
  _AppStatsStruct *m_elem_stats_ptr;
};


struct _AppStats
{
  Array<_AppStatsStruct> m_query_stats;
  Array<_AppStatsStruct> m_elem_stats;

  _AppStatsAccess get_host_appstats()
  {
    return { m_query_stats.get_host_ptr(),
             m_elem_stats.get_host_ptr() };
  }

  _AppStatsAccess get_device_appstats()
  {
    return { m_query_stats.get_device_ptr(),
             m_elem_stats.get_device_ptr() };
  }

  static bool is_enabled() { return true; }
};


// Empty definitions.
struct NullAppStatsStruct { };
struct NullAppStatsAccess { };
struct NullAppStats
{
  NullAppStatsAccess get_host_appstats()   { NullAppStatsAccess ret; return ret; }
  NullAppStatsAccess get_device_appstats() { NullAppStatsAccess ret; return ret; }

  static bool is_enabled() { return false; }
};


#ifdef DRAY_STATS
  using AppStatsStruct = _AppStatsStruct;
  using AppStatsAccess = _AppStatsAccess;
  using AppStats = _AppStats;
#else
  using AppStatsStruct = NullAppStatsStruct;
  using AppStatsAccess = NullAppStatsAccess;
  using AppStats = NullAppStats;
#endif//DRAY_STATS


  extern GlobalShare<AppStats> global_app_stats;
} // namespace stats
} // namespace dray

#endif//DRAY_APPSTATS_HPP

#ifndef DRAY_STATS_HPP
#define DRAY_STATS_HPP

#include <dray/types.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>
#include <RAJA/RAJA.hpp>

namespace dray
{


template <typename T, int32 mult>
struct _MultiReduceSum : public _MultiReduceSum<T,mult-1>
{
  RAJA::ReduceSum<reduce_policy,T> m_count = RAJA::ReduceSum<reduce_policy,T>(0);

  DRAY_EXEC RAJA::ReduceSum<reduce_policy,T> & operator[] (int ii)
  {
    return *(&m_count - mult + 1 + ii);
  }

  void get(T sums[])
  {
    for (int ii = 0; ii < mult; ii++)
    {
      sums[ii] = static_cast<T>(operator[](ii).get());
    }
  }
};
template <typename T>
struct _MultiReduceSum<T,0> {};

template <typename T, int32 nbins>
struct HistogramSmall
{
  const T *m_sep;  //[nbins-1];   // User must initialize the separator values and this pointer to array T[nbins-1].
  _MultiReduceSum<int32,nbins> m_count;

  DRAY_EXEC void datum(T x)
  {
    // Iterate over all bins until find the bin for x.
    int32 b = 0;
    while (b < nbins-1 && x > m_sep[b])
    {
      b++;
    }

    // Increment bin for x.
    m_count[b] += 1;
  }

  void get(T sums[]) { m_count.get(sums); }

  static void log(const T *sep, const int32 *counts)
  {
    DRAY_LOG_OPEN("histogram_small");
    int32 ii;
    for (ii = 0; ii < nbins - 1; ii++)
    {
      DRAY_LOG_ENTRY("hist_bin_count", counts[ii]);
      DRAY_LOG_ENTRY("hist_bin_r_sep", sep[ii]);
    }
    DRAY_LOG_ENTRY("hist_bin_count", counts[ii]);
    DRAY_LOG_CLOSE();
  }
};

////// //
////// // Stats - Class to accumulate statistical counters using RAJA.
////// //
////// template <typename T>   // T must have comparison operators defined.
////// struct Stats
////// {
//////   // abilities:
//////   // - min
//////   // - max
//////   //
//////   // - small_histogram     // Static arrays in registers -> multiple reduce
//////   // - mid_histogram       // External memory -> sweep and atomic
//////   // - large_histogram     // External memory -> binary search and atomic
////// 
//////   // The first and last bins extend to their respective infinities.
//////   // The set of bins is defined by bin separators. The bin to the left of any
//////   // separator contains <= the separator value; the bin to the right of any
//////   // separator contains > the separator value.
//////   //
//////   // The separators array is read-only to this class (TODO take advantage of this for optimizations).
//////   // The hist array must be writable by this class.
////// 
//////   static Stats factory_small_histogram(T min_val, T max_val, 
////// 
////// };

}  // namespace dray

#endif // DRAY_STATS_HPP

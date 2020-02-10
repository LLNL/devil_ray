#include "dray/sort.hpp"

#include <algorithm>

#include <dray/array_utils.hpp>
#include <dray/bvh_utils.hpp>

#if defined(DRAY_CUDA_ENABLED)

// NOTE: uses the cub installation that is bundled with RAJA
#include "cub/device/device_radix_sort.cuh"

#elif defined(USE_OPENMP)

#include <omp.h>

#endif

namespace dray {

Array<int32> sort(Array<uint32> &mcodes)
{
  const size_t size = mcodes.size ();
  Array<int32> iter = array_counting (size, 0, 1);
  int32 *iter_ptr = iter.get_device_ptr ();
  uint32 *mcodes_ptr = mcodes.get_device_ptr ();

#if defined(DRAY_CUDA_ENABLED)
  // the case where we do have CUDA enabled

  uint32 *mcodes_alt_buf = nullptr;
  cudaMalloc(&mcodes_alt_buf, size*(sizeof(uint32)));

  int32* iter_alt_buf = nullptr;
  cudaMalloc(&iter_alt_buf, size*(sizeof(int32)));

  // create double buffers
  ::cub::DoubleBuffer< uint32 > d_keys( mcodes_ptr, mcodes_alt_buf );
  ::cub::DoubleBuffer< int32 >  d_values( iter_ptr, iter_alt_buf );

  // determine temporary device storage requirements
  void * d_temp_storage     = nullptr;
  size_t temp_storage_bytes = 0;
  ::cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes,
                                   d_keys, d_values, size );

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);


  // Run sorting operation
  cudaErrchk(::cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes,
                                     d_keys, d_values, size ));

  uint32* sorted_keys = d_keys.Current();
  int32*  sorted_vals = d_values.Current();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (size_t i)
  {
    mcodes_ptr[i] = sorted_keys[i];
    iter_ptr[i]   = sorted_vals[i];
  });

  // Free temporary storage
  cudaFree(d_temp_storage);
  cudaFree(mcodes_alt_buf);
  cudaFree(iter_alt_buf);

//https://gist.github.com/wanghc78/2c2b403299cab172e74c62f4397a6997
#elif defined(USE_OPENMP) // no CUDA but we have OpenMP

  #define BASE_BITS 8
  #define BASE (1 << BASE_BITS)
  #define MASK (BASE-1)
  #define DIGITS(v, shift) (((v) >> shift) & MASK)

  size_t n = size;
  int32 *data = iter_ptr;
  int32 *buffer = (int32 *)malloc(n*sizeof(int32));
  int total_digits = sizeof(int32)*8;

  // Each thread use local_bucket to move data
  size_t i;
  for (int shift = 0; shift < total_digits; shift+=BASE_BITS)
  {
  size_t bucket[BASE] = {0};

  size_t local_bucket[BASE] = {0}; // size needed in each bucket/thread
  // 1st pass, scan whole and check the count
  #pragma omp parallel firstprivate(local_bucket)
  {
    #pragma omp for schedule(static) nowait
    for (i = 0; i < n; ++i)
    {
      local_bucket[DIGITS(mcodes_ptr[data[i]], shift)]++;
    }
    #pragma omp critical
    for (i = 0; i < BASE; ++i)
    {
      bucket[i] += local_bucket[i];
    }
    #pragma omp barrier
    #pragma omp single
    for (i = 1; i < BASE; ++i)
    {
      bucket[i] += bucket[i - 1];
    }
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    for (int cur_t = nthreads - 1; cur_t >= 0; cur_t--)
    {
      if(cur_t == tid)
      {
        for (i = 0; i < BASE; ++i)
        {
          bucket[i] -= local_bucket[i];
          local_bucket[i] = bucket[i];
        }
      } else
      { // just do barrier
        #pragma omp barrier
      }

    }
    #pragma omp for schedule(static)
    for (i = 0; i < n; ++i)
    { // note here the end condition
      buffer[local_bucket[DIGITS(mcodes_ptr[data[i]], shift)]++] = data[i];
    }
  }
    // now move data
    int32* tmp = data;
    data = buffer;
    buffer = tmp;
  }

  free(buffer);

  reorder(iter, mcodes);

#else
  // the case where we have neither CUDA nor OpenMP

  std::sort(iter_ptr,
            iter_ptr + size,
            [=](int32 i1, int32 i2)
            {
              return mcodes_ptr[i1] < mcodes_ptr[i2];
            });

  reorder(iter, mcodes);

#endif

  return iter;
}

} // namespace dray

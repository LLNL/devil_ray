#ifndef DRAY_SORT_HPP
#define DRAY_SORT_HPP

#warning "included"

#include <algorithm>
#include <iostream>

#include <dray/array_utils.hpp>
#include <dray/bvh_utils.hpp>

// FIXME: remove this when header guard created
//#define DRAY_OPENMP_ENABLED

#if defined(DRAY_CUDA_ENABLED)
// NOTE: uses the cub installation that is bundled with RAJA
#include "cub/device/device_radix_sort.cuh"

#elif defined(DRAY_OPENMP_ENABLED)
#include <omp.h>

#endif

namespace dray {

Array<int32> sort(Array<uint32> &mcodes)
{
  const int size = mcodes.size ();
  Array<int32> iter = array_counting (size, 0, 1);
  // TODO: create custom sort for GPU / CPU
  int32 *iter_ptr = iter.get_host_ptr ();
  uint32 *mcodes_ptr = mcodes.get_host_ptr ();

#if defined(DRAY_CUDA_ENABLED)
#warning "cuda"
  // the case where we do have CUDA enabled

  Array<uint32> mcodes_alt;
  mcodes_alt.resize(size);
  uint32 *mcodes_alt_buf = mcodes_alt.get_device_ptr();
  // uint32 *mcodes_alt_buf = nullptr;
  // cudaMalloc(&mcodes_alt_buf, size);

  Array<int32> iter_alt;
  iter_alt.resize(size);
  int32* iter_alt_buf = iter_alt.get_device_ptr();
  //int32* iter_alt_buf = nullptr;
  //cudaMalloc(&iter_alt_buf, size);

  // // create double buffers
  ::cub::DoubleBuffer< uint32 > d_keys( mcodes_ptr, mcodes_alt_buf );
  ::cub::DoubleBuffer< int32 >  d_values( iter_ptr, iter_alt_buf );

  // determine temporary device storage requirements
  void * d_temp_storage     = nullptr;
  size_t temp_storage_bytes = 0;
  ::cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes,
                                   d_keys, d_values, size );


  // Allocate temporary storage
  Array<unsigned char> d_temp_storage_buf;
  d_temp_storage_buf.resize(temp_storage_bytes);
  d_temp_storage = (void*)d_temp_storage_buf.get_device_ptr();
  //cudaMalloc(&d_temp_storage, temp_storage_bytes);


  // Run sorting operation
  ::cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes,
                                     d_keys, d_values, size );

  uint32* sorted_keys = d_keys.Current();
  int32*  sorted_vals = d_values.Current();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    mcodes_ptr[i] = sorted_keys[i];
    iter_ptr[i]   = sorted_vals[i];
  });

  // // Free temporary storage
  // FIXME: does something need to be done here?
  // axom::deallocate( d_temp_storage );
  // axom::deallocate( mcodes_alt_buf );
  // axom::deallocate( iter_alt_buf );
  // cudaFree(&mcodes_alt_buf);
  // cudaFree(&iter_alt_buf);
  // cudaFree(&d_temp_storage);

//https://gist.github.com/wanghc78/2c2b403299cab172e74c62f4397a6997
#elif defined(DRAY_OPENMP_ENABLED) // no CUDA but we have OpenMP
#warning "openmp"

  #define BASE_BITS 8
  #define BASE (1 << BASE_BITS)
  #define MASK (BASE-1)
  #define DIGITS(v, shift) (((v) >> shift) & MASK)

  size_t n = size;
  int32 *data = iter_ptr;
  int32 *buffer = (int32 *)malloc(n*sizeof(int32));
  int total_digits = sizeof(int32)*8;

  //Each thread use local_bucket to move data
  size_t i;
  for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {
  size_t bucket[BASE] = {0};

  size_t local_bucket[BASE] = {0}; // size needed in each bucket/thread
  //1st pass, scan whole and check the count
  #pragma omp parallel firstprivate(local_bucket)
  {
      #pragma omp for schedule(static) nowait
      for(i = 0; i < n; i++){
  	local_bucket[DIGITS(mcodes_ptr[data[i]], shift)]++;
      }
      #pragma omp critical
      for(i = 0; i < BASE; i++) {
  	bucket[i] += local_bucket[i];
      }
      #pragma omp barrier
      #pragma omp single
      for (i = 1; i < BASE; i++) {
  	bucket[i] += bucket[i - 1];
      }
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();
      for(int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
  	if(cur_t == tid) {
  	    for(i = 0; i < BASE; i++) {
  		bucket[i] -= local_bucket[i];
  		local_bucket[i] = bucket[i];
  	    }
  	} else { //just do barrier
  	    #pragma omp barrier
  	}

      }
      #pragma omp for schedule(static)
      for(i = 0; i < n; i++) { //note here the end condition
  	buffer[local_bucket[DIGITS(mcodes_ptr[data[i]], shift)]++] = data[i];
      }
    }
    //now move data
    int32* tmp = data;
    data = buffer;
    buffer = tmp;
  }

  free(buffer);

#else
#warning "std"
  // the case where we have neither CUDA nor OpenMP

  std::sort(iter_ptr,
            iter_ptr + size,
            [=](int32 i1, int32 i2)
            {
              return mcodes_ptr[i1] < mcodes_ptr[i2];
            });

#endif

  reorder(iter, mcodes);

  return iter;
}

} // namespace dray

#endif // DRAY_SORT_HPP

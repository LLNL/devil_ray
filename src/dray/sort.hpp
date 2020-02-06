#ifndef DRAY_SORT_HPP
#define DRAY_SORT_HPP

#warning "included"

//#include <algorithm>
//#include <iostream>

#include <dray/array_utils.hpp>
#include <dray/bvh_utils.hpp>

//// FIXME: remove this when header guard created
////#define DRAY_OPENMP_ENABLED
//
//#if defined(DRAY_CUDA_ENABLED)
//// NOTE: uses the cub installation that is bundled with RAJA
//#include "cub/device/device_radix_sort.cuh"
//
//#elif defined(DRAY_OPENMP_ENABLED)
//#include <omp.h>

namespace dray {

Array<int32> sort(Array<uint32> &mcodes);

} // namespace dray

#endif // DRAY_SORT_HPP

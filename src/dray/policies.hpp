#ifndef DRAY_POLICIECS_HPP
#define DRAY_POLICIECS_HPP

#include <dray/exports.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{

//using for_policy = RAJA::omp_parallel_for_exec;
//using reduce_policy = RAJA::omp_reduce;

///using for_policy = RAJA::seq_exec;
///using reduce_policy = RAJA::seq_reduce;
#define BLOCK_SIZE 128
using for_policy = RAJA::cuda_exec<BLOCK_SIZE>;
using reduce_policy = RAJA::cuda_reduce<BLOCK_SIZE>;

} // namespace dray
#endif

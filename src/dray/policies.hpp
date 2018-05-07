#ifndef DRAY_POLICIECS_HPP
#define DRAY_POLICIECS_HPP

#include <RAJA/RAJA.hpp>

namespace dray
{

//using for_policy = RAJA::omp_parallel_for_exec;
//using reduce_policy = RAJA::omp_reduce;

using for_policy = RAJA::seq_exec;
using reduce_policy = RAJA::seq_reduce;

} // namespace dray
#endif

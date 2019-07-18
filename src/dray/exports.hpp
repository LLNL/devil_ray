#ifndef DRAY_EXPORTS_HPP
#define DRAY_EXPORTS_HPP

#if defined(__CUDACC__) && ! defined(DEBUG_CPU_ONLY)

#define DRAY_CUDA_ENABLED
#define DRAY_EXEC inline __host__ __device__
#define DRAY_EXEC_ONLY inline __device__
#define DRAY_LAMBDA __device__

#else

#define DRAY_EXEC __inline__
#define DRAY_EXEC_ONLY __inline__
#define DRAY_LAMBDA

#endif

#endif

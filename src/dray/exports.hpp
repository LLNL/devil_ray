#ifndef DRAY_EXPORTS_HPP
#define DRAY_EXPORTS_HPP

#if defined __CUDACC__

#define CUDA_ENABLED 
#define DRAY_EXEC inline __host__ __device__
#define DRAY_EXEC_ONLY inline __device__
#define DRAY_LAMBDA __device__

#else

#define DRAY_EXEC inline
#define DRAY_EXEC_ONLY inline
#define DRAY_LAMBDA  

#endif

#endif

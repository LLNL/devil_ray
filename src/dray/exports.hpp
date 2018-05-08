#ifndef DRAY_EXPORTS_HPP
#define DRAY_EXPORTS_HPP

#if defined __CUDACC__
#define DRAY_EXEC inline __host__ __device__
#else
#define DRAY_EXEC inline
#endif
#define DRAY_LAMBDA __device__

#endif

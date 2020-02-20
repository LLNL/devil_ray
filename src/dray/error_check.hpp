// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ERROR_CHECK_HPP
#define DRAY_ERROR_CHECK_HPP

#include <dray/exports.hpp>
#include <iostream>

namespace dray
{
#ifdef DRAY_CUDA_ENABLED
inline void cuda_error_check(const char *file, const int line )
{
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    std::cerr<<"CUDA error reported at: "<<file<<":"<<line;
    std::cerr<<" : "<<cudaGetErrorString(err)<<"\n";
    //exit( -1 );
  }
}
#define DRAY_ERROR_CHECK() cuda_error_check(__FILE__,__LINE__);
#else
#define DRAY_ERROR_CHECK()
#endif

} // namespace dray
#endif

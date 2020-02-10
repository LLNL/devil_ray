// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"
#include <dray/array.hpp>

//#define DRAY_CUDA_ENABLED

#include <dray/sort.hpp>

TEST (dray_sort, dray_sort_test)
{
  unsigned int toSort[] = {0, 7, 1, 6, 2, 5, 3, 4};
  dray::Array<unsigned int> array;

  array.set(toSort, 8);

  dray::Array<int> iter = dray::sort(array);


  unsigned int *sorted = array.get_host_ptr();
  int *sorted_iter = iter.get_host_ptr();

  for (size_t i = 0; i < 8; ++i)
    std::cout << sorted[i] << std::endl;

  for (size_t i = 0; i < 8; ++i)
    std::cout << sorted_iter[i] << std::endl;

}

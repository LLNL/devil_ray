// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"
#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/bvh_utils.hpp>
#include <dray/types.hpp>

//#define DRAY_CUDA_ENABLED

#include <dray/sort.hpp>

using namespace dray;

// reference sort implementation
//dray::Array<int> std_sort(dray::Array<unsigned int> &mcodes)
//{
//  const size_t size = mcodes.size ();
//  dray::Array<int> iter = dray::array_counting (size, 0, 1);
//  int *iter_ptr = iter.get_host_ptr ();
//  unsigned int *mcodes_ptr = mcodes.get_host_ptr ();
//
//  std::sort(iter_ptr,
//            iter_ptr + size,
//            [=](int i1, int i2)
//            {
//              return mcodes_ptr[i1] < mcodes_ptr[i2];
//            });
//
//  reorder(iter, mcodes);
//
//  return iter;
//}


void std_sort(Array<uint32> &mcodes)
{
  const size_t size = mcodes.size ();
  uint32 *mcodes_ptr = mcodes.get_host_ptr ();

  std::sort(mcodes_ptr, mcodes_ptr + size);

}

TEST (dray_sort, dray_sort_test)
{

  const size_t LARGE_NUM = 9999999;

  srand(2);
  unsigned int *mcodes = new unsigned int[LARGE_NUM];

  for (size_t i = 0; i < LARGE_NUM; ++i)
  {
      mcodes[i] = std::rand();
  }

  //dray::Array<unsigned int> a2;
  //a2.set(mcodes, LARGE_NUM);
  //std_sort(a2);
  //unsigned int *a2_ptr = a2.get_host_ptr();

  dray::Array<unsigned int> a1;
  a1.set(mcodes, LARGE_NUM);
  dray::Array<int> iter = dray::sort(a1);
  unsigned int *a1_ptr = a1.get_host_ptr();
  int *iter_ptr = iter.get_host_ptr();

  //for (size_t i = 0; i < LARGE_NUM; ++i)
  //{
  //  ASSERT_EQ(a1_ptr[i], a2_ptr[i]);
  //}

  for (size_t i = 0; i < LARGE_NUM; ++i)
  {
    ASSERT_EQ(mcodes[iter_ptr[i]], a1_ptr[i]);
  }



  //unsigned int toSort[] = {0, 7, 1, 6, 2, 5, 3, 4};
  //dray::Array<unsigned int> array;

  //array.set(toSort, 8);

  //dray::Array<int> iter = dray::sort(array);


  //unsigned int *sorted = array.get_host_ptr();
  //int *sorted_iter = iter.get_host_ptr();

  //for (size_t i = 0; i < 8; ++i)
  //  std::cout << sorted[i] << std::endl;

  //for (size_t i = 0; i < 8; ++i)
  //  std::cout << sorted_iter[i] << std::endl;

}

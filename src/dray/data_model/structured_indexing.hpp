// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_STRUCTURED_INDEXING_HPP
#define DRAY_STRUCTURED_INDEXING_HPP

#include <dray/exports.hpp>
#include <dray/location.hpp>
#include <dray/math.hpp>
#include <dray/vec.hpp>

namespace dray
{

DRAY_EXEC
Vec<int32,3>
logical_index_3d(const int32 index,
                 const Vec<int32,3> &dims)
{
  Vec<int32,3> idx;
  idx[0] = index % dims[0];
  idx[1] = (index / dims[0]) % dims[1];
  idx[2] = index / (dims[0] * dims[1]);
  return idx;
}

DRAY_EXEC
Vec<int32,3>
logical_index_2d(const int32 index, const Vec<int32,3> &dims)
{
  Vec<int32,3> idx;
  idx[0] = index % dims[0];
  idx[1] = index / dims[0];
  return idx;
}

DRAY_EXEC
int32
flat_index_2d(const Vec<int32,2> &idx, const Vec<int32,2> &dims)
{
  return idx[1] * dims[0] + idx[0];
}

DRAY_EXEC
int32
flat_index_3d(const Vec<int32,3> &idx, const Vec<int32,3> &dims)
{
  return (idx[2] * dims[1] + idx[1] ) * dims[0] + idx[0];
}

//DRAY_EXEC
//Float interpolate_3d(const Location, )
//{
//  return (idx[2] * dims[1] + idx[1] ) * dims[0] + idx[0];
//}


} // namespace dray

#endif // DRAY_TOPOLGY_BASE_HPP

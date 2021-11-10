// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/uniform_indexer.hpp>

#include <unordered_set>


using UI = dray::UniformIndexer;
using AllCells = UI::AllCells;
using AllFaces = UI::AllFaces;
using AllVerts = UI::AllVerts;
using SideFaces = UI::SideFaces;
using SideVerts = UI::SideVerts;

// -----------------------------------------------------------------------------
// Define specializations of std::hash and std::equal_to
// to store POD types in std::unordered_map.
// -----------------------------------------------------------------------------

#define HASH_STRUCT(class_name, obj_name, body)\
template <> struct std::hash<class_name>\
{\
  size_t operator()(const class_name & obj_name) const noexcept\
  {\
    body \
  }\
};

#define EQUAL_TO_STRUCT(class_name, lhs_name, rhs_name, body)\
template <> struct std::equal_to<class_name>\
{\
  bool operator()(const class_name &lhs_name, const class_name &rhs_name) const\
  {\
    body \
  }\
};

inline size_t interleave(std::initializer_list<int> x);  // interleave bits

HASH_STRUCT(AllCells, v, { return interleave({v.idx[0], v.idx[1], v.idx[2]}); });
HASH_STRUCT(AllFaces, v, { return interleave({int(v.plane), v.idx[0], v.idx[1], v.idx[2]}); });
HASH_STRUCT(AllVerts, v, { return interleave({v.idx[0], v.idx[1], v.idx[2]}); });
HASH_STRUCT(SideFaces, v, { return interleave({int(v.side), v.idx[0], v.idx[1], v.idx[2]}); });
HASH_STRUCT(SideVerts, v, { return interleave({int(v.side), v.idx[0], v.idx[1], v.idx[2]}); });

EQUAL_TO_STRUCT(AllCells, lhs, rhs, { return lhs.idx == rhs.idx; });
EQUAL_TO_STRUCT(AllFaces, lhs, rhs, { return lhs.plane == rhs.plane  &&  lhs.idx == rhs.idx; });
EQUAL_TO_STRUCT(AllVerts, lhs, rhs, { return lhs.idx == rhs.idx; });
EQUAL_TO_STRUCT(SideFaces, lhs, rhs, { return lhs.side == rhs.side  &&  lhs.idx == rhs.idx; });
EQUAL_TO_STRUCT(SideVerts, lhs, rhs, { return lhs.side == rhs.side  &&  lhs.idx == rhs.idx; });

#undef HASH_STRUCT
#undef EQUAL_TO_STRUCT


// -----------------------------------------------------------------------------
// Unit tests
// -----------------------------------------------------------------------------

static const dray::Vec<int, 3> cell_dims = {{16, 4, 8}};

using RAJA::forall;
using RAJA::seq_exec;
using RAJA::RangeSegment;
using RAJA::ReduceSum;

//
// Uniqueness
//

TEST (dray_uniform_indexer, uniq_all_cells)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 size = idxr.all_cells_size();
  std::unordered_set<AllCells> uniq;
  forall<seq_exec>(RangeSegment(0, size), [=, &uniq] DRAY_LAMBDA (int32 i)
  {
    uniq.insert(idxr.all_cells(i));
  });
  EXPECT_TRUE(uniq.size() == size);
}

TEST (dray_uniform_indexer, uniq_all_verts)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 size = idxr.all_verts_size();
  std::unordered_set<AllVerts> uniq;
  forall<seq_exec>(RangeSegment(0, size), [=, &uniq] DRAY_LAMBDA (int32 i)
  {
    uniq.insert(idxr.all_verts(i));
  });
  EXPECT_TRUE(uniq.size() == size);
}

TEST (dray_uniform_indexer, uniq_all_faces)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 size = idxr.all_faces_size();
  std::unordered_set<AllFaces> uniq;
  forall<seq_exec>(RangeSegment(0, size), [=, &uniq] DRAY_LAMBDA (int32 i)
  {
    uniq.insert(idxr.all_faces(i));
  });
  EXPECT_TRUE(uniq.size() == size);
}

TEST (dray_uniform_indexer, uniq_side_faces)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const int32 size = idxr.side_faces_size({side});
    std::unordered_set<SideFaces> uniq;
    forall<seq_exec>(RangeSegment(0, size), [=, &uniq] DRAY_LAMBDA (int32 i)
    {
      uniq.insert(idxr.side_faces(side, i));
    });
    EXPECT_TRUE(uniq.size() == size);
  }
}

TEST (dray_uniform_indexer, uniq_side_verts)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const int32 size = idxr.side_verts_size({side});
    std::unordered_set<SideVerts> uniq;
    forall<seq_exec>(RangeSegment(0, size), [=, &uniq] DRAY_LAMBDA (int32 i)
    {
      uniq.insert(idxr.side_verts(side, i));
    });
    EXPECT_TRUE(uniq.size() == size);
  }
}


//
// Invertibility
//

TEST (dray_uniform_indexer, inv_all_cells)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 size = idxr.all_cells_size();
  ReduceSum<reduce_policy, int32> failures(0);
  forall<for_policy>(RangeSegment(0, size), [=, &failures] DRAY_LAMBDA (int32 i)
  {
    const bool success = (i == idxr.flat_idx(idxr.all_cells(i)));
    failures += !success;
  });
  EXPECT_TRUE(failures.get() == 0);
}

TEST (dray_uniform_indexer, inv_all_verts)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 size = idxr.all_verts_size();
  ReduceSum<reduce_policy, int32> failures(0);
  forall<for_policy>(RangeSegment(0, size), [=, &failures] DRAY_LAMBDA (int32 i)
  {
    const bool success = (i == idxr.flat_idx(idxr.all_verts(i)));
    failures += !success;
  });
  EXPECT_TRUE(failures.get() == 0);
}

TEST (dray_uniform_indexer, inv_all_faces)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 size = idxr.all_faces_size();
  ReduceSum<reduce_policy, int32> failures(0);
  forall<for_policy>(RangeSegment(0, size), [=, &failures] DRAY_LAMBDA (int32 i)
  {
    const bool success = (i == idxr.flat_idx(idxr.all_faces(i)));
    failures += !success;
  });
  EXPECT_TRUE(failures.get() == 0);
}

TEST (dray_uniform_indexer, inv_side_faces)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const int32 size = idxr.side_faces_size({side});
    ReduceSum<reduce_policy, int32> failures(0);
    forall<for_policy>(RangeSegment(0, size), [=, &failures] DRAY_LAMBDA (int32 i)
    {
      const bool success = (i == idxr.flat_idx(idxr.side_faces(side, i)));
      failures += !success;
    });
    EXPECT_TRUE(failures.get() == 0);
  }
}

TEST (dray_uniform_indexer, inv_side_verts)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const int32 size = idxr.side_verts_size({side});
    ReduceSum<reduce_policy, int32> failures(0);
    forall<for_policy>(RangeSegment(0, size), [=, &failures] DRAY_LAMBDA (int32 i)
    {
      const bool success = (i == idxr.flat_idx(idxr.side_verts(side, i)));
      failures += !success;
    });
    EXPECT_TRUE(failures.get() == 0);
  }
}


//
// Adjacency
//

//TODO


// ------------------------------

inline size_t interleave(std::initializer_list<int> x)
{
  //     e  e  e | e  e  e|  e  e  
  //      e  e  e|  e  e  |e  e  e
  //       e  e  |e  e  e | e  e  e

  const size_t nbits = 8 * sizeof(size_t);
  const size_t shift_mask = nbits - 1;
  const size_t nx = x.size();

  size_t z = 0;
  size_t shift = 0;
  for (size_t b = 0; b < nbits; ++b)
    for (const int xi : x)
    {
      const size_t bit = (size_t(xi) >> b) & 1u;
      z ^= (bit << shift);
      shift = (shift + 1) & shift_mask;
    }
  return z;
}



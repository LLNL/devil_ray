// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/uniform_indexer.hpp>
#include <dray/array_utils.hpp>
#include <dray/host_array.hpp>

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
  EXPECT_EQ(uniq.size(), size);
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
  EXPECT_EQ(uniq.size(), size);
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
  EXPECT_EQ(uniq.size(), size);
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
    EXPECT_EQ(uniq.size(), size);
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
    EXPECT_EQ(uniq.size(), size);
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
  EXPECT_EQ(failures.get(), 0);
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
  EXPECT_EQ(failures.get(), 0);
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
  EXPECT_EQ(failures.get(), 0);
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
    EXPECT_EQ(failures.get(), 0);
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
    EXPECT_EQ(failures.get(), 0);
  }
}


//
// Adjacency
//

std::array<dray::UniformIndexer::Side, dray::UniformIndexer::NUM_SIDES> mirror_map()
{
  using UI = dray::UniformIndexer;
  std::array<UI::Side, UI::NUM_SIDES> mirror;
  mirror[UI::Z0] = UI::Z1;
  mirror[UI::Z1] = UI::Z0;
  mirror[UI::Y0] = UI::Y1;
  mirror[UI::Y1] = UI::Y0;
  mirror[UI::X0] = UI::X1;
  mirror[UI::X1] = UI::X0;
  return mirror;
}

TEST (dray_uniform_indexer, side_faces_mirror)
{
  using namespace dray;
  using UI = UniformIndexer;
  UniformIndexer idxr = {cell_dims};

  std::array<UI::Side, UI::NUM_SIDES> mirror = mirror_map();

  for (int s = 0; s < 6; ++s)
  {
    const UI::Side side = UI::side(s);
    const UI::Side mirror_side = mirror[side];
    const int32 size = idxr.side_faces_size({side});
    ReduceSum<reduce_policy, int32> failures(0);
    forall<for_policy>(RangeSegment(0, size), [=, &failures] DRAY_LAMBDA (int32 i)
    {
      const Vec<int32, 2> idx = sub_vec<2>(
          idxr.side_faces(side, i).idx,
          idxr.axis_subset(side));
      const Vec<int32, 2> mirror_idx = sub_vec<2>(
          idxr.side_faces(mirror_side, i).idx,
          idxr.axis_subset(mirror_side));

      const bool success = (idx == mirror_idx);
      failures += !success;
    });
    EXPECT_EQ(failures.get(), 0);
  }
}

TEST (dray_uniform_indexer, side_verts_mirror)
{
  using namespace dray;
  using UI = UniformIndexer;
  UniformIndexer idxr = {cell_dims};

  std::array<UI::Side, UI::NUM_SIDES> mirror = mirror_map();

  for (int s = 0; s < 6; ++s)
  {
    const UI::Side side = UI::side(s);
    const UI::Side mirror_side = mirror[side];
    const int32 size = idxr.side_verts_size({side});
    ReduceSum<reduce_policy, int32> failures(0);
    forall<for_policy>(RangeSegment(0, size), [=, &failures] DRAY_LAMBDA (int32 i)
    {
      const Vec<int32, 2> idx = sub_vec<2>(
          idxr.side_verts(side, i).idx,
          idxr.axis_subset(side));
      const Vec<int32, 2> mirror_idx = sub_vec<2>(
          idxr.side_verts(mirror_side, i).idx,
          idxr.axis_subset(mirror_side));

      const bool success = (idx == mirror_idx);
      failures += !success;
    });
    EXPECT_EQ(failures.get(), 0);
  }
}

TEST (dray_uniform_indexer, normal_mirror)
{
  using namespace dray;
  using UI = UniformIndexer;

  std::array<UI::Side, UI::NUM_SIDES> mirror = mirror_map();

  for (int s = 0; s < 6; ++s)
  {
    const UI::Side side = UI::side(s);
    const UI::Side mirror_side = mirror[side];
    EXPECT_EQ(UI::normal(side), -UI::normal(mirror_side));
  }
}


//future: cells-faces, faces-verts


//
// Subset
//

TEST (dray_uniform_indexer, inv_gather_scatter_verts)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const UniformIndexer::SideVertSet side_set = idxr.side_vert_set(side);
    const int32 sub_size = idxr.side_verts_size(side_set);
    const int32 size = idxr.all_verts_size();
    Array<int32> src = array_counting(sub_size, 0, 1);
    Array<int32> big = array_val(std::numeric_limits<int32>::max(), size);
    Array<int32> dest = array_val(-size, sub_size);

    idxr.scatter(side_set, src, big);
    idxr.gather(side_set, dest, big);

    const int32 diff = array_max_diff(src, dest);
    EXPECT_EQ(0, diff);
  }
}

TEST (dray_uniform_indexer, inv_gather_scatter_faces)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const UniformIndexer::SideFaceSet side_set = idxr.side_face_set(side);
    const int32 sub_size = idxr.side_faces_size(side_set);
    const int32 size = idxr.all_faces_size();
    Array<int32> src = array_counting(sub_size, 0, 1);
    Array<int32> big = array_val(std::numeric_limits<int32>::max(), size);
    Array<int32> dest = array_val(-size, sub_size);

    idxr.scatter(side_set, src, big);
    idxr.gather(side_set, dest, big);

    const int32 diff = array_max_diff(src, dest);
    EXPECT_EQ(0, diff);
  }
}

TEST (dray_uniform_indexer, count_overlap_side_verts)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 all_size = idxr.all_verts_size();

  Array<int32> meet = array_zero<int32>(all_size);

  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const UniformIndexer::SideVertSet side_set = idxr.side_vert_set(side);
    const int32 sub_size = idxr.side_verts_size(side_set);

    Array<int32> side_array;
    side_array.resize(sub_size);
    idxr.gather(side_set, side_array, meet);
    side_array = array_map(side_array,
        [=] DRAY_LAMBDA (int32 x) { return x + 1; });
    idxr.scatter(side_set, side_array, meet);
  }

  Array<int32> interior_points = index_where(meet, 0);
  Array<int32> face_points = index_where(meet, 1);
  Array<int32> edge_points = index_where(meet, 2);
  Array<int32> corner_points = index_where(meet, 3);

  Vec<int32, 3> c = cell_dims;
  c[0]--;
  c[1]--;
  c[2]--;

  EXPECT_EQ( c[0] * c[1] * c[2],  interior_points.size() );
  EXPECT_EQ( 2*c[0]*c[1] + 2*c[0]*c[2] + 2*c[1]*c[2],  face_points.size() );
  EXPECT_EQ( 4*c[0] + 4*c[1] + 4*c[2],  edge_points.size() );
  EXPECT_EQ( 8,  corner_points.size() );
}

TEST (dray_uniform_indexer, loc_overlap_side_verts)
{
  using namespace dray;
  UniformIndexer idxr = {cell_dims};
  const int32 all_size = idxr.all_verts_size();

  Array<int32> meet = array_zero<int32>(all_size);

  for (int s = 0; s < 6; ++s)
  {
    const UniformIndexer::Side side = UniformIndexer::side(s);
    const UniformIndexer::SideVertSet side_set = idxr.side_vert_set(side);
    const int32 sub_size = idxr.side_verts_size(side_set);

    Array<int32> side_array;
    side_array.resize(sub_size);
    idxr.gather(side_set, side_array, meet);
    side_array = array_map(side_array,
        [=] DRAY_LAMBDA (int32 x) { return x + 1; });
    idxr.scatter(side_set, side_array, meet);
  }

  Array<int32> corner_points = index_where(meet, 3);
  ConstHostArray<int32> h_corner_points(corner_points);
  for (int32 i = 0; i < corner_points.size(); ++i)
  {
    const int32 flat = h_corner_points.get_item(i);
    const UniformIndexer::AllVerts all_verts = idxr.all_verts(flat);

    const Vec<int32, 3> expand_i = {{ bool(i & (1u << 0)),
                                      bool(i & (1u << 1)),
                                      bool(i & (1u << 2)) }};
    const Vec<int32, 3> expected_idx = hadamard(expand_i, cell_dims);

    EXPECT_EQ(expected_idx, all_verts.idx);
  }
}


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



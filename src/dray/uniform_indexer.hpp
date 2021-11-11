// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_INDEXER_HPP
#define DRAY_UNIFORM_INDEXER_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>

#include <dray/policies.hpp>
#include <dray/exports.hpp>
#include <dray/device_array.hpp>
/// #include <dray/array_utils.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{
  // Note: No attempt has yet been made to utilize RAJA iteration
  //       spaces to avoid read-collisions in GPU global memory.

  struct UniformIndexer  // TODO merge with UniformFaces
  {
    public:
      enum Side : uint8 { Z0 = 0, Z1, Y0, Y1, X0, X1, NUM_SIDES };
      enum Plane : uint8 { Z, Y, X, NUM_PLANES };
      static Side side(uint8 side) { return static_cast<Side>(side); }
      static Plane plane(uint8 plane) { return static_cast<Plane>(plane); }
      using FlatIdx = int32;

      // Unit within the set of all units
      struct AllCells;
      struct AllFaces;
      struct AllVerts;

      // Unit within the subset of units on a side of the mesh.
      struct SideFaces;
      struct SideVerts;

      // Subset of units on a side of the mesh.
      struct SideFaceSet;
      struct SideVertSet;

      DRAY_EXEC static Vec<uint8, 3> axis_subset(const Side side);
      DRAY_EXEC static Vec<uint8, 3> axis_subset(const Plane plane);

      DRAY_EXEC SideFaceSet side_face_set(const Side side) const;
      DRAY_EXEC SideVertSet side_vert_set(const Side side) const;

      DRAY_EXEC AllCells all_cells(const FlatIdx flat_idx) const;
      DRAY_EXEC AllFaces all_faces(const FlatIdx flat_idx) const;
      DRAY_EXEC AllVerts all_verts(const FlatIdx flat_idx) const;
      DRAY_EXEC SideFaces side_faces(const Side side, const FlatIdx flat_idx) const;
      DRAY_EXEC SideVerts side_verts(const Side side, const FlatIdx flat_idx) const;

      DRAY_EXEC FlatIdx flat_idx(const AllCells &all_cells) const;
      DRAY_EXEC FlatIdx flat_idx(const AllFaces &all_faces) const;
      DRAY_EXEC FlatIdx flat_idx(const AllVerts &all_verts) const;
      DRAY_EXEC FlatIdx flat_idx(const SideFaces &side_faces) const;
      DRAY_EXEC FlatIdx flat_idx(const SideVerts &side_verts) const;

      // Adjacency mappings for fields of different associations
      // (faces on a cell, vertices on a face)
      DRAY_EXEC AllFaces all_faces(const AllCells &all_cells, const Side side) const;
      DRAY_EXEC AllVerts all_verts(const AllFaces &all_faces, const int32 corner) const;
      DRAY_EXEC SideVerts side_verts(const SideFaces &side_faces, const int32 corner) const;

      // Conversion of individual indices for scatter and gather.
      DRAY_EXEC AllFaces all_faces(const SideFaces &side_faces) const;
      DRAY_EXEC AllVerts all_verts(const SideVerts &side_verts) const;

      // Logical normals, for world normals must multiply by world parity.
      DRAY_EXEC static Vec<int32, 3> normal(const Side side);
      DRAY_EXEC static Vec<int32, 3> normal(const Plane plane);

      // Gather (copy) and scatter (overwrite) in mesh side vs total array.
      template <typename T>
      void gather(const SideFaceSet side_set,
                  Array<T> &side_faces,
                  const Array<T> &total_faces) const;
      template <typename T>
      void scatter(const SideFaceSet side_set,
                   const Array<T> &side_faces,
                   Array<T> &total_faces) const;

      template <typename T>
      void gather(const SideVertSet side_set,
                  Array<T> &side_verts,
                  const Array<T> &total_verts) const;
      template <typename T>
      void scatter(const SideVertSet side_set,
                   const Array<T> &side_verts,
                   Array<T> &total_verts) const;

      // Expected sizes of sets for vertex-/face-associated arrays.
      size_t all_cells_size() const;
      size_t all_faces_size() const;
      size_t all_verts_size() const;
      size_t side_faces_size(const SideFaceSet side_set) const;
      size_t side_verts_size(const SideVertSet side_set) const;

      // ------------------------------------------

      Vec<int32, 3> m_cell_dims;

    private:
      const Vec<int32, 3> & dims() const { return m_cell_dims; }
      const Vec<int32, 3>   Dims() const { return m_cell_dims +
                                                  Vec<int32, 3>{{1,1,1}}; }

      DRAY_EXEC int32 dims_x() const { return m_cell_dims[0]; }
      DRAY_EXEC int32 dims_y() const { return m_cell_dims[1]; }
      DRAY_EXEC int32 dims_z() const { return m_cell_dims[2]; }
      DRAY_EXEC int32 Dims_x() const { return m_cell_dims[0] + 1; }
      DRAY_EXEC int32 Dims_y() const { return m_cell_dims[1] + 1; }
      DRAY_EXEC int32 Dims_z() const { return m_cell_dims[2] + 1; }
  };

  struct UniformIndexer::AllCells
  {
    Vec<int32, 3> idx;
  };

  struct UniformIndexer::AllFaces
  {
    Plane plane;
    Vec<int32, 3> idx;
  };

  struct UniformIndexer::AllVerts
  {
    Vec<int32, 3> idx;
  };

  struct UniformIndexer::SideFaces
  {
    Side side;
    Vec<int32, 3> idx;
  };

  struct UniformIndexer::SideVerts
  {
    Side side;
    Vec<int32, 3> idx;
  };

  struct UniformIndexer::SideFaceSet
  {
    Side side;
  };

  struct UniformIndexer::SideVertSet
  {
    Side side;
  };
}



// implementations
namespace dray
{
  // axis_subset(Side)
  DRAY_EXEC Vec<uint8, 3> UniformIndexer::axis_subset(const Side side)
  {
    Vec<uint8, 3> axis_subset = {{true, true, true}};
    if (side == Z0 || side == Z1)  axis_subset[2] = false;
    if (side == Y0 || side == Y1)  axis_subset[1] = false;
    if (side == X0 || side == X1)  axis_subset[0] = false;
    return axis_subset;
  }

  // axis_subset(Plane)
  DRAY_EXEC Vec<uint8, 3> UniformIndexer::axis_subset(const Plane plane)
  {
    Vec<uint8, 3> axis_subset = {{true, true, true}};
    if (plane == Z)  axis_subset[2] = false;
    if (plane == Y)  axis_subset[1] = false;
    if (plane == X)  axis_subset[0] = false;
    return axis_subset;
  }

  DRAY_EXEC UniformIndexer::SideFaceSet
  UniformIndexer::side_face_set(const Side side) const
  {
    return SideFaceSet{side};
  }

  DRAY_EXEC UniformIndexer::SideVertSet
  UniformIndexer::side_vert_set(const Side side) const
  {
    return SideVertSet{side};
  }

  // all_cells_size()
  size_t UniformIndexer::all_cells_size() const
  {
    return dims_x() * dims_y() * dims_z();
  }

  // all_faces_size()
  size_t UniformIndexer::all_faces_size() const
  {
    return dims_x() * dims_y() * Dims_z() +
           dims_x() * Dims_y() * dims_z() +
           Dims_x() * dims_y() * dims_z();
  }

  // all_verts_size()
  size_t UniformIndexer::all_verts_size() const
  {
    return Dims_x() * Dims_y() * Dims_z();
  }

  // side_faces_size()
  size_t UniformIndexer::side_faces_size(const SideFaceSet side_set) const
  {
    const Vec<uint8, 3> axis_subset = this->axis_subset(side_set.side);
    const Vec<int32, 2> dims = sub_vec<2>(this->dims(), axis_subset);
    return dims[0] * dims[1];
  }

  // side_verts_size()
  size_t UniformIndexer::side_verts_size(const SideVertSet side_set) const
  {
    const Vec<uint8, 3> axis_subset = this->axis_subset(side_set.side);
    const Vec<int32, 2> Dims = sub_vec<2>(this->Dims(), axis_subset);
    return Dims[0] * Dims[1];
  }

  // all_cells()
  DRAY_EXEC UniformIndexer::AllCells
  UniformIndexer::all_cells(const FlatIdx flat_idx) const
  {
    Vec<int32, 3> idx;
    idx[0] = flat_idx % dims_x();
    idx[1] = (flat_idx / dims_x()) % dims_y();
    idx[2] = flat_idx / (dims_x() * dims_y());
    return AllCells{idx};
  }

  // all_verts()
  DRAY_EXEC UniformIndexer::AllVerts
  UniformIndexer::all_verts(const FlatIdx flat_idx) const
  {
    Vec<int32, 3> idx;
    idx[0] = flat_idx % Dims_x();
    idx[1] = (flat_idx / Dims_x()) % Dims_y();
    idx[2] = flat_idx / (Dims_x() * Dims_y());
    return AllVerts{idx};
  }

  // all_faces()
  DRAY_EXEC UniformIndexer::AllFaces
  UniformIndexer::all_faces(const FlatIdx flat_idx) const
  {
    const int32 offset_z = dims_x() * dims_y() * Dims_z();
    const int32 offset_y = dims_x() * dims_z() * Dims_y();

    const Plane plane = (flat_idx < offset_z ?             Z
                      :  flat_idx < offset_z + offset_y ?  Y
                      :                                    X);
    const int32 offset = (plane == Z ?   0
                       :  plane == Y ?   offset_z
                       :/*plane == X ?*/ offset_z + offset_y);

    const int32 flat = flat_idx - offset;

    Vec<int32, 3> face_dims = dims();
    if (plane == Z) face_dims[2] += 1;
    if (plane == Y) face_dims[1] += 1;
    if (plane == X) face_dims[0] += 1;

    Vec<int32, 3> idx;
    idx[0] = flat % face_dims[0];
    idx[1] = (flat / face_dims[0]) % face_dims[1];
    idx[2] = flat / (face_dims[0] * face_dims[1]);

    return AllFaces{plane, idx};
  }

  // side_faces()
  DRAY_EXEC UniformIndexer::SideFaces
  UniformIndexer::side_faces(const Side side, const FlatIdx flat_idx) const
  {
    const Vec<uint8, 3> axis_subset = this->axis_subset(side);
    const Vec<int32, 2> dims = sub_vec<2>(this->dims(), axis_subset);

    Vec<int32, 2> idx;
    idx[0] = flat_idx % dims[0];
    idx[1] = flat_idx / dims[0];

    Vec<int32, 3> idx3 = super_vec<3>(idx, axis_subset, 0);
    if (side == Z1)  idx3[2] = dims_z();
    if (side == Y1)  idx3[1] = dims_y();
    if (side == X1)  idx3[0] = dims_x();

    return SideFaces{side, idx3};
  }

  // side_verts()
  DRAY_EXEC UniformIndexer::SideVerts
  UniformIndexer::side_verts(const Side side, const FlatIdx flat_idx) const
  {
    const Vec<uint8, 3> axis_subset = this->axis_subset(side);
    const Vec<int32, 2> Dims = sub_vec<2>(this->Dims(), axis_subset);

    Vec<int32, 2> idx;
    idx[0] = flat_idx % Dims[0];
    idx[1] = flat_idx / Dims[0];

    Vec<int32, 3> idx3 = super_vec<3>(idx, axis_subset, 0);
    if (side == Z1)  idx3[2] = dims_z();
    if (side == Y1)  idx3[1] = dims_y();
    if (side == X1)  idx3[0] = dims_x();

    return SideVerts{side, idx3};
  }

  // flat_idx(AllCells)
  DRAY_EXEC UniformIndexer::FlatIdx
  UniformIndexer::flat_idx(const AllCells &all_cells) const
  {
    const Vec<int32, 3> &idx = all_cells.idx;
    return idx[0] + dims_x() * (idx[1] + dims_y() * (idx[2]));
  }

  // flat_idx(AllVerts)
  DRAY_EXEC UniformIndexer::FlatIdx
  UniformIndexer::flat_idx(const AllVerts &all_verts) const
  {
    // Dims == dims + 1
    const Vec<int32, 3> &idx = all_verts.idx;
    return idx[0] + Dims_x() * (idx[1] + Dims_y() * (idx[2]));
  }

  // flat_idx(AllFaces)
  DRAY_EXEC UniformIndexer::FlatIdx
  UniformIndexer::flat_idx(const AllFaces &all_faces) const
  {
    // Dims == dims + 1
    const Plane plane = all_faces.plane;
    const Vec<int32, 3> &idx = all_faces.idx;

    const int32 offset_z = dims_x() * dims_y() * Dims_z();
    const int32 offset_y = dims_x() * dims_z() * Dims_y();
    const int32 offset = (plane == Z ?   0
                       :  plane == Y ?   offset_z
                       :/*plane == X ?*/ offset_z + offset_y);

    FlatIdx flat;
    if (plane == Z)
      flat = idx[0] + dims_x() * (idx[1] + dims_y() * (idx[2]));
    if (plane == Y)
      flat = idx[0] + dims_x() * (idx[1] + Dims_y() * (idx[2]));
    if (plane == X)
      flat = idx[0] + Dims_x() * (idx[1] + dims_y() * (idx[2]));

    return FlatIdx(offset + flat);
  }

  // flat_idx(SideFaces)
  DRAY_EXEC UniformIndexer::FlatIdx
  UniformIndexer::flat_idx(const SideFaces &side_faces) const
  {
    const Side side = side_faces.side;
    const Vec<uint8, 3> axis_subset = this->axis_subset(side);

    const Vec<int32, 2> idx = sub_vec<2>(side_faces.idx, axis_subset);
    const Vec<int32, 2> dims = sub_vec<2>(this->dims(), axis_subset);

    return idx[0] + dims[0] * (idx[1]);
  }

  // flat_idx(SideVerts)
  DRAY_EXEC UniformIndexer::FlatIdx
  UniformIndexer::flat_idx(const SideVerts &side_verts) const
  {
    const Side side = side_verts.side;
    const Vec<uint8, 3> axis_subset = this->axis_subset(side);

    const Vec<int32, 2> idx = sub_vec<2>(side_verts.idx, axis_subset);
    const Vec<int32, 2> Dims = sub_vec<2>(this->Dims(), axis_subset);

    return idx[0] + Dims[0] * (idx[1]);
  }

  // all_faces(AllCells)
  DRAY_EXEC UniformIndexer::AllFaces
  UniformIndexer::all_faces(const AllCells &all_cells, const Side side) const
  {
    Vec<int32, 3> idx = all_cells.idx;
    Plane plane;
    if (side == Z0)  { plane = Z; }
    if (side == Z1)  { plane = Z;  idx[2] += 1; }
    if (side == Y0)  { plane = Y; }
    if (side == Y1)  { plane = Y;  idx[1] += 1; }
    if (side == X0)  { plane = X; }
    if (side == X1)  { plane = X;  idx[0] += 1; }

    return AllFaces{plane, idx};
  }

  // all_verts(AllFaces)
  DRAY_EXEC UniformIndexer::AllVerts
  UniformIndexer::all_verts(const AllFaces &all_faces, const int32 corner) const
  {
    Vec<int32, 3> idx = all_faces.idx;
    const Plane plane = all_faces.plane;

    // Expect: 0 <= corner < 4
    Vec<int32, 2> inc = {{ bool(corner & (1 << 0)),
                           bool(corner & (1 << 1)) }};

    if (plane == Z)  { idx[0] += inc[0];  idx[1] += inc[1]; }
    if (plane == Y)  { idx[0] += inc[0];  idx[2] += inc[1]; }
    if (plane == X)  { idx[1] += inc[0];  idx[2] += inc[1]; }

    return AllVerts{idx};
  }

  // side_verts(SideFaces)
  DRAY_EXEC UniformIndexer::SideVerts
  UniformIndexer::side_verts(const SideFaces &side_faces, const int32 corner) const
  {
    Vec<int32, 3> idx = side_faces.idx;
    const Side side = side_faces.side;

    // Expect: 0 <= corner < 4
    Vec<int32, 2> inc = {{ bool(corner & (1 << 0)),
                           bool(corner & (1 << 1)) }};

    if (side == Z0 || side == Z1)  { idx[0] += inc[0];  idx[1] += inc[1]; }
    if (side == Y0 || side == Y1)  { idx[0] += inc[0];  idx[2] += inc[1]; }
    if (side == X0 || side == X1)  { idx[1] += inc[0];  idx[2] += inc[1]; }

    return SideVerts{side, idx};
  }

  // all_faces(side_faces)
  DRAY_EXEC UniformIndexer::AllFaces
  UniformIndexer::all_faces(const SideFaces &side_faces) const
  {
    const Side side = side_faces.side;
    const Plane plane = (side == Z0 || side == Z1 ? Z
                      :  side == Y0 || side == Y1 ? Y
                      :/*side == X0 || side == X1*/ X);

    return AllFaces{plane, side_faces.idx};
  }

  // all_verts(side_verts)
  DRAY_EXEC UniformIndexer::AllVerts
  UniformIndexer::all_verts(const SideVerts &side_verts) const
  {
    return AllVerts{side_verts.idx};
  }

  DRAY_EXEC Vec<int32, 3> UniformIndexer::normal(const Side side)
  {
    int32 sign = 1;
    if (side == Z0 || side == Y0 || side == X0)
      sign = -1;

    Vec<int32, 2> sub_vec = {{0, 0}};
    Vec<int32, 3> vec = super_vec<3>(sub_vec, axis_subset(side), sign);
    return vec;
  }

  DRAY_EXEC Vec<int32, 3> UniformIndexer::normal(const Plane plane)
  {
    int32 sign = 1;

    Vec<int32, 2> sub_vec = {{0, 0}};
    Vec<int32, 3> vec = super_vec<3>(sub_vec, axis_subset(plane), sign);
    return vec;
  }

  // gather(): SideFaces <- TotalFaces
  template <typename T>
  void UniformIndexer::gather(
      const SideFaceSet side_set,
      Array<T> &side_faces,
      const Array<T> &total_faces) const
  {
    const size_t subset_size = this->side_faces_size(side_set);
    const size_t total_size = this->all_faces_size();

    assert(side_faces.size() == subset_size);
    assert(total_faces.size() == total_size);

    const UniformIndexer idxr = *this;
    const Side side = side_set.side;
    NonConstDeviceArray<T> d_side_faces(side_faces);
    ConstDeviceArray<T> d_total_faces(total_faces);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, subset_size),
        [=] DRAY_LAMBDA (int32 sub_i)
    {
      const SideFaces side_faces = idxr.side_faces(side, sub_i);
      const AllFaces all_faces = idxr.all_faces(side_faces);
      const int32 i = idxr.flat_idx(all_faces);

      d_side_faces.get_item(sub_i) = d_total_faces.get_item(i);
    });
  }

  // scatter(): SideFaces -> TotalFaces
  template <typename T>
  void UniformIndexer::scatter(
      const SideFaceSet side_set,
      const Array<T> &side_faces,
      Array<T> &total_faces) const
  {
    const size_t subset_size = this->side_faces_size(side_set);
    const size_t total_size = this->all_faces_size();

    assert(side_faces.size() == subset_size);
    assert(total_faces.size() == total_size);

    const UniformIndexer idxr = *this;
    const Side side = side_set.side;
    ConstDeviceArray<T> d_side_faces(side_faces);
    NonConstDeviceArray<T> d_total_faces(total_faces);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, subset_size),
        [=] DRAY_LAMBDA (int32 sub_i)
    {
      const SideFaces side_faces = idxr.side_faces(side, sub_i);
      const AllFaces all_faces = idxr.all_faces(side_faces);
      const int32 i = idxr.flat_idx(all_faces);

      d_total_faces.get_item(i) = d_side_faces.get_item(sub_i);
    });
  }

  // gather(): SideVerts <- TotalVerts
  template <typename T>
  void UniformIndexer::gather(
      const SideVertSet side_set,
      Array<T> &side_verts,
      const Array<T> &total_verts) const
  {
    const size_t subset_size = this->side_verts_size(side_set);
    const size_t total_size = this->all_verts_size();

    assert(side_verts.size() == subset_size);
    assert(total_verts.size() == total_size);

    const UniformIndexer idxr = *this;
    const Side side = side_set.side;
    NonConstDeviceArray<T> d_side_verts(side_verts);
    ConstDeviceArray<T> d_total_verts(total_verts);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, subset_size),
        [=] DRAY_LAMBDA (int32 sub_i)
    {
      const SideVerts side_verts = idxr.side_verts(side, sub_i);
      const AllVerts all_verts = idxr.all_verts(side_verts);
      const int32 i = idxr.flat_idx(all_verts);

      d_side_verts.get_item(sub_i) = d_total_verts.get_item(i);
    });
  }

  // scatter(): SideVerts -> TotalVerts
  template <typename T>
  void UniformIndexer::scatter(
      const SideVertSet side_set,
      const Array<T> &side_verts,
      Array<T> &total_verts) const
  {
    const size_t subset_size = this->side_verts_size(side_set);
    const size_t total_size = this->all_verts_size();

    assert(side_verts.size() == subset_size);
    assert(total_verts.size() == total_size);

    const UniformIndexer idxr = *this;
    const Side side = side_set.side;
    ConstDeviceArray<T> d_side_verts(side_verts);
    NonConstDeviceArray<T> d_total_verts(total_verts);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, subset_size),
        [=] DRAY_LAMBDA (int32 sub_i)
    {
      const SideVerts side_verts = idxr.side_verts(side, sub_i);
      const AllVerts all_verts = idxr.all_verts(side_verts);
      const int32 i = idxr.flat_idx(all_verts);

      d_total_verts.get_item(i) = d_side_verts.get_item(sub_i);
    });
  }
}

#endif//DRAY_UNIFORM_INDEXER_HPP

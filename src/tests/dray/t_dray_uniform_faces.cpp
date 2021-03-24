// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include <RAJA/RAJA.hpp>

#include <dray/types.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/uniform_faces.hpp>


static dray::Array<dray::Vec<dray::Float, 3>>
  gen_cell_centers(dray::UniformTopology &topo);


TEST(dray_uniform_faces, dray_uniform_faces)
{
  using dray::Vec;
  using dray::Float;
  using dray::int32;
  using dray::uint8;

  const Vec<Float, 3> origin = {{0.f, 0.f, 0.f}};
  const Vec<Float, 3> spacing = {{11.f/4.f,  13.f/8.f,  17.f/16.f}};
  const Vec<int32, 3> cell_dims = {{3, 7, 11}};
  fprintf(stdout, "origin==[%f, %f, %f];  spacing==[%f, %f, %f];  dims==[%d, %d, %d]\n",
      origin[0], origin[1], origin[2],
      spacing[0], spacing[1], spacing[2],
      cell_dims[0], cell_dims[1], cell_dims[2]);

  dray::UniformTopology uni_topo(spacing, origin, cell_dims);

  // Cell centers.
  dray::Array<Vec<Float, 3>> cell_centers = gen_cell_centers(uni_topo);

  // Cell faces.
  dray::UniformFaces uni_faces = dray::UniformFaces::from_uniform_topo(uni_topo);
  dray::Array<Vec<Float, 3>> face_centers;
  face_centers.resize(uni_faces.num_total_faces());
  uni_faces.fill_total_faces(face_centers.get_host_ptr());

  // Compare.
  int32 num_cells = cell_centers.size();
  int32 num_faces = face_centers.size();
  fprintf(stdout, "num_cells==%d;  num_faces==%d\n", num_cells, num_faces);
  for (int ii = 0; ii < num_cells; ++ii)
  {
    const Vec<Float, 3> cell_center = cell_centers.get_host_ptr_const()[ii];

    using FaceID = dray::UniformFaces::FaceID;
    for (uint8 face = 0; face < FaceID::NUM_FACES; ++face)
    {
      Vec<Float, 3> anticipated = cell_center;
      if (face == FaceID::X0)
        anticipated[0] -= spacing[0] * 0.5;
      if (face == FaceID::X1)
        anticipated[0] += spacing[0] * 0.5;
      if (face == FaceID::Y0)
        anticipated[1] -= spacing[1] * 0.5;
      if (face == FaceID::Y1)
        anticipated[1] += spacing[1] * 0.5;
      if (face == FaceID::Z0)
        anticipated[2] -= spacing[2] * 0.5;
      if (face == FaceID::Z1)
        anticipated[2] += spacing[2] * 0.5;

      const int32 face_idx = uni_faces.cell_idx_to_face_idx(ii, FaceID(face));
      Vec<Float, 3> actual = face_centers.get_host_ptr_const()[face_idx];

      EXPECT_EQ(anticipated, actual);
    }
  }
}




static dray::Array<dray::Vec<dray::Float,3>> gen_cell_centers(dray::UniformTopology &topo)
{
  using dray::Vec;
  using dray::int32;
  using dray::Float;
  using dray::Array;
  using dray::for_policy;

  const Vec<int32,3> cell_dims = topo.cell_dims();
  const Vec<Float,3> origin = topo.origin();
  const Vec<Float,3> spacing = topo.spacing();

  const int32 num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];

  Array<Vec<Float,3>> locations;
  locations.resize(num_cells);
  Vec<Float,3> *loc_ptr = locations.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_cells), [=] DRAY_LAMBDA (int32 index)
  {
    Vec<int32,3> cell_id;
    cell_id[0] = index % cell_dims[0];
    cell_id[1] = (index / cell_dims[0]) % cell_dims[1];
    cell_id[2] = index / (cell_dims[0] * cell_dims[1]);

    Vec<Float,3> loc;
    for(int32 i = 0; i < 3; ++i)
    {
      loc[i] = origin[i] + Float(cell_id[i]) * spacing[i] + spacing[i] * 0.5f;
    }

    loc_ptr[index] = loc;
  });

  return locations;
}

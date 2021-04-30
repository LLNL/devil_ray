// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/data_model/low_order_field.hpp>
#include <dray/array_utils.hpp>
#include <dray/uniform_faces.hpp>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <iostream>

dray::Float analytical_f(const dray::Vec<dray::Float, 3> &xyz);

template <typename F_Vec3d_to_scalar>
void add_vertex_field(
    dray::DataSet &dataset, const std::string &name, const F_Vec3d_to_scalar &f);

void add_interpolated_vertex_field(
    dray::DataSet &src_dataset, const std::string &src_name,
    dray::DataSet &dst_dataset, const std::string &dst_name);

dray::Float linf_vertex_fields(
    dray::DataSet &dataset, const std::string &name_a, const std::string &name_b);

TEST(t_dray_lerp_uniform_vert, refine)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "lerp_uni");

  using dray::Vec;
  using dray::Float;
  int coarse_elems_1d = 10;
  double coarse_spacing_1d = 1.0/coarse_elems_1d;
  Vec<int, 3> cell_dims = (Vec<int, 3> {{1,1,1}}) * coarse_elems_1d;
  Vec<Float, 3> spacing = (Vec<Float, 3> {{1.f,1.f,1.f}}) * coarse_spacing_1d;
  Vec<Float, 3> origin = {{0.f, 0.f, 0.f}};

  // coarse_mesh and fine_mesh
  std::shared_ptr<dray::UniformTopology> coarse_mesh
    = std::make_shared<dray::UniformTopology>(
        spacing, origin, cell_dims);
  std::shared_ptr<dray::UniformTopology> fine_mesh
    = std::make_shared<dray::UniformTopology>(
        spacing * 0.5, origin, cell_dims * 2);
  coarse_mesh->name("topo");
  fine_mesh->name("topo");

  // coarse_dataset and fine_dataset
  dray::DataSet coarse_dataset(coarse_mesh);
  dray::DataSet fine_dataset(fine_mesh);

  add_vertex_field(coarse_dataset, "coarse_field", analytical_f);
  /// {
  ///   // output to blueprint
  ///   conduit::Node bp_dataset;
  ///   coarse_dataset.to_blueprint(bp_dataset);
  ///   conduit::relay::io::blueprint::save_mesh(bp_dataset, output_file + "_c.blueprint_root_hdf5");
  /// }

  add_interpolated_vertex_field(coarse_dataset, "coarse_field",
                                fine_dataset, "interp_field");

  add_vertex_field(fine_dataset, "fine_field", analytical_f);
    ///conduit::relay::io::blueprint::save_mesh(bp_dataset, output_file + "_f.blueprint_root_hdf5");

  Float diff = linf_vertex_fields(fine_dataset, "fine_field", "interp_field");
  printf("interpolated error on fine field == %f\n", diff);
  EXPECT_FLOAT_EQ(0.0, diff);
}


dray::Float analytical_f(const dray::Vec<dray::Float, 3> &xyz)
{
  const dray::Float &x = xyz[0], &y = xyz[1], &z = xyz[2];
  return (x*x + (1.-y)*(1.-y) + (1.-z)*z);
}

template <typename F_Vec3d_to_scalar>
void add_vertex_field(
    dray::DataSet &dataset, const std::string &name, const F_Vec3d_to_scalar &f)
{
  // mesh
  dray::UniformTopology * mesh;
  if (!(mesh = dynamic_cast<dray::UniformTopology*>(dataset.mesh())))
  {
    fprintf(stderr, "Expected a uniform mesh.\n");
    exit(-1);
  }
  const dray::Vec<dray::int32, 3> dims = mesh->cell_dims();
  const dray::Vec<dray::Float, 3> spacing = mesh->spacing();
  const dray::Vec<dray::Float, 3> origin = mesh->origin();

  // values
  dray::Array<dray::Float> values;
  values.resize((dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1));
  dray::Float *values_p = values.get_host_ptr();
  dray::int32 index = 0;
  for (int k = 0; k < dims[2] + 1; ++k)
  {
    const dray::Float z = origin[2] + spacing[2] * k;
    for (int j = 0; j < dims[1] + 1; ++j)
    {
      const dray::Float y = origin[1] + spacing[1] * j;
      for (int i = 0; i < dims[0] + 1; ++i)
      {
        const dray::Float x = origin[0] + spacing[0] * i;
        const dray::Float v = f(dray::Vec<dray::Float, 3>{{x, y, z}});
        values_p[index] = v;
        index++;
      }
    }
  }

  // field and dataset
  std::shared_ptr<dray::LowOrderField> field
    = std::make_shared<dray::LowOrderField>(values, dray::LowOrderField::Assoc::Vertex);
  field->name(name);
  dataset.add_field(field);
}

void add_interpolated_vertex_field(
    dray::DataSet &src_dataset, const std::string &src_name,
    dray::DataSet &dst_dataset, const std::string &dst_name)
{
  // mesh
  dray::UniformTopology * dst_mesh;
  if (!(dst_mesh = dynamic_cast<dray::UniformTopology*>(dst_dataset.mesh())))
  {
    fprintf(stderr, "Expected a uniform mesh.");
    exit(-1);
  }
  const dray::Vec<dray::int32, 3> dims = dst_mesh->cell_dims();
  const dray::Vec<dray::Float, 3> spacing = dst_mesh->spacing();
  const dray::Vec<dray::Float, 3> origin = dst_mesh->origin();

  // values
  dray::Array<dray::Float> values;
  values.resize((dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1));
  dray::array_memset_zero(values);
  dray::Float *values_p = values.get_host_ptr();

  // vert_coords
  dray::Array<dray::Vec<dray::Float, 3>> vert_coords;
  vert_coords.resize((dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1));
  dray::Vec<dray::Float, 3> * vert_coords_p = vert_coords.get_host_ptr();
  dray::int32 index = 0;
  for (int k = 0; k < dims[2] + 1; ++k)
  {
    const dray::Float z = origin[2] + spacing[2] * k;
    for (int j = 0; j < dims[1] + 1; ++j)
    {
      const dray::Float y = origin[1] + spacing[1] * j;
      for (int i = 0; i < dims[0] + 1; ++i)
      {
        const dray::Float x = origin[0] + spacing[0] * i;
        vert_coords_p[index] = {{x, y, z}};
        index++;
      }
    }
  }

  // interpolate from coarse mesh at the fine mesh vertices
  dray::Array<dray::Location> locations = src_dataset.mesh()->locate(vert_coords);
  {
    // Check that locate always succeeds;
    // in this unit test, the bounds on both meshes should be equal.
    int32 failures = 0;
    for (int32 index = 0; index < locations.size(); ++index)
      if (locations.get_host_ptr()[index].m_cell_id == -1)
        failures++;
    if (failures > 0)
      fprintf(stderr, "Locate failed on %d of %d points.\n",
          failures, locations.size());
    EXPECT_EQ(failures, 0);
  }
  src_dataset.field(src_name)->eval(locations, values);

  // dst_field and dst_dataset
  std::shared_ptr<dray::LowOrderField> dst_field
    = std::make_shared<dray::LowOrderField>(values, dray::LowOrderField::Assoc::Vertex);
  dst_field->name(dst_name);
  dst_dataset.add_field(dst_field);
}

dray::Float linf_vertex_fields(
    dray::DataSet &dataset, const std::string &name_a, const std::string &name_b)
{
  // fields
  dray::LowOrderField * field_a;
  dray::LowOrderField * field_b;
  if (!(field_a = dynamic_cast<dray::LowOrderField *>(dataset.field(name_a))))
  {
    fprintf(stderr, "Expected field \"%s\" to be a uniform low-order field.\n",
        name_a.c_str());
    exit(-1);
  }
  if (!(field_b = dynamic_cast<dray::LowOrderField *>(dataset.field(name_b))))
  {
    fprintf(stderr, "Expected field \"%s\" to be a uniform low-order field.\n",
        name_b.c_str());
    exit(-1);
  }

  // check same association and size
  if (field_a->assoc() != field_b->assoc())
  {
    fprintf(stderr, "Expected fields \"%s\" and \"%s\" to match association\n",
        name_a.c_str(), name_b.c_str());
    exit(-1);
  }
  if (field_a->values().size() != field_b->values().size())
  {
    fprintf(stderr, "Expected fields \"%s\" and \"%s\" to match size\n",
        name_a.c_str(), name_b.c_str());
    exit(-1);
  }

  // compare values
  dray::Float linf_diff = 0;
  dray::int32 size = field_a->values().size();
  const dray::Float * field_a_p = field_a->values().get_host_ptr_const();
  const dray::Float * field_b_p = field_b->values().get_host_ptr_const();
  for (dray::int32 index = 0; index < size; ++index)
  {
    dray::Float diff = abs(field_a_p[index] - field_b_p[index]);
    if (linf_diff < diff)
      linf_diff = diff;
  }

  return linf_diff;
}




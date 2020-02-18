// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <vector>
#include <random>
#include <algorithm>

///#include <dray/io/blueprint_reader.hpp>
#include <dray/math.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/Element/element.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/grid_function.hpp>
#include <dray/data_set.hpp>
#include <dray/derived_topology.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_degree_raising, dray_degree_raising_3_5)
{
  // Scalar field over the reference square.
  constexpr int ncomp = 3;
  constexpr int dim = 3;

  constexpr int deg_raise = 2;

  const int p_lo = 3;
  const int p_hi = p_lo + deg_raise;

  using ElementType = dray::Element<dim, ncomp, dray::ElemType::Quad, dray::Order::General>;
  /// using ElementType = dray::Element<dim, ncomp, dray::ElemType::Tri, dray::Order::General>;
  // Commented out triangle/tet version until triangles/test are activated.

  const int npe_lo = ElementType::get_num_dofs(p_lo);
  const int npe_hi = ElementType::get_num_dofs(p_hi);

  const int num_elems = 5;

  using dray::Float;
  using DofT = dray::Vec<Float, ncomp>;

  std::vector<Float> host_data_lo(ncomp * num_elems * npe_lo, 0);
  std::vector<Float> host_data_hi(ncomp * num_elems * npe_hi, 0);

  std::vector<int> host_mapping_lo(num_elems * npe_lo);
  std::vector<int> host_mapping_hi(num_elems * npe_hi);
  std::iota(host_mapping_lo.begin(), host_mapping_lo.end(), 0);
  std::iota(host_mapping_hi.begin(), host_mapping_hi.end(), 0);

  // Fill with random data.
  // TODO start with flat data and perturb it.
  // TODO new test that uses the furnace interface to load a read dataset.
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{10,2};
  for (int nidx = 0; nidx < ncomp * num_elems * npe_lo; nidx++)
    host_data_lo[nidx] = d(gen);

  std::vector<ElementType> host_elem_collection_lo(num_elems);
  std::vector<ElementType> host_elem_collection_hi(num_elems);

  // Project each element from lo basis to hi basis representation.
  for (int eid = 0; eid < num_elems; eid++)
  {
    dray::ReadDofPtr<dray::Vec<Float, ncomp>> read_lo{&host_mapping_lo[eid * npe_lo], (DofT*) &host_data_lo[0]};
    dray::WriteDofPtr<dray::Vec<Float, ncomp>> write_hi{&host_mapping_hi[eid * npe_hi], (DofT*) &host_data_hi[0]};

    ElementType &elem_lo = host_elem_collection_lo[eid];
    ElementType &elem_hi = host_elem_collection_hi[eid];
    elem_lo.construct(eid, read_lo, p_lo);
    elem_hi.construct(eid, write_hi.to_readonly_dof_ptr(), p_hi);

    ElementType::template project_to_higher_order_basis<deg_raise>(elem_lo, elem_hi, write_hi);
  }

  // Create a grid of samples in the reference square.
  constexpr int gridres = 4;
  std::vector<dray::Vec<Float, dim>> samples((gridres+1)*(gridres+1)*(gridres+1));
  for (int kk = 0; kk <= gridres; kk++)
    for (int jj = 0; jj <= gridres; jj++)
      for (int ii = 0; ii <= gridres; ii++)
        samples[kk*(gridres+1)*(gridres+1) + jj*(gridres+1) + ii] = \
            dray::Vec<Float, dim>{ 1.0f / gridres * ii, 1.0f / gridres * jj, 1.0f / gridres * kk };

  // Test by making sure that evaluation is unchanged in the new basis.
  for (int eid = 0; eid < num_elems; eid++)
  {
    for (int sid = 0; sid < (gridres+1)*(gridres+1)*(gridres+1); sid++)
    {
      if (ElementType::is_inside(samples[sid]))
      {
        dray::Vec<DofT, dim> unused_jac;
        DofT value_lo = host_elem_collection_lo[eid].eval_d(samples[sid], unused_jac);
        DofT value_hi = host_elem_collection_hi[eid].eval_d(samples[sid], unused_jac);
        for (int c = 0; c < ncomp; ++c)
        {
          EXPECT_FLOAT_EQ(value_lo[c], value_hi[c]);
        }
        /// std::cout << value_lo << " ==? " << value_hi << "\n";
      }
    }
  }

  // GridFunction->Mesh->Topology->DataSet
  dray::GridFunction<3u> mesh_data_hi;
  mesh_data_hi.m_el_dofs = npe_hi;
  mesh_data_hi.m_size_el = num_elems;
  mesh_data_hi.m_size_ctrl = npe_hi * num_elems;
  mesh_data_hi.m_values = dray::Array<DofT>((DofT*) &(*host_data_hi.begin()), host_data_hi.size()/3);
  mesh_data_hi.m_ctrl_idx = dray::Array<int>(&(*host_mapping_hi.begin()), host_mapping_hi.size());

  dray::Mesh<ElementType> mesh_hi(mesh_data_hi, p_hi);
  ///DerivedTopology<ElementType> topo_hi(mesh_hi);
  dray::DataSet data_set_hi(std::make_shared<dray::DerivedTopology<ElementType>>(mesh_hi));
  /// for (const std::string &field_name : data_set.fields())
  ///   data_set_hi.add_field(std::shared_ptr<FieldBase>(data_set.field(field_name)));
  std::cout << "Made a dataset.\n";
}

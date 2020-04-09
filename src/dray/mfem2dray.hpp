// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MFEM2DRAY_HPP
#define DRAY_MFEM2DRAY_HPP

#include <mfem.hpp>

#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/grid_function.hpp>
#include <dray/GridFunction/mesh.hpp>

#include <dray/import_order_policy.hpp>
#include <dray/data_set.hpp>

namespace dray
{

DataSet import_mesh(const mfem::Mesh &mesh,
                    const ImportOrderPolicy &import_order_policy);

void import_field(DataSet &dataset,
                  const ImportOrderPolicy &import_order_policy,
                  const mfem::GridFunction &grid_function,
                  const mfem::Geometry::Type geom_type,
                  const std::string field_name,
                  const int32 comp = -1);

//
// project_to_pos_basis()
//
// Helper function prototype.
// If is_new was set to true, the caller is responsible for deleting the returned pointer.
// If is_new was set to false, then the returned value is null, and the caller should use gf.
mfem::GridFunction *project_to_pos_basis (const mfem::GridFunction *gf, bool &is_new);

} // namespace dray

#endif // DRAY_MFEM2DRAY_HPP

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BLUEPRINT_READER_HPP
#define DRAY_BLUEPRINT_READER_HPP

#include <conduit.hpp>
#include <dray/data_set.hpp>
#include <dray/import_order_policy.hpp>

namespace dray
{

class BlueprintReader
{
  public:
  static DataSet load (const std::string &root_file, const int cycle, const ImportOrderPolicy &);

  static DataSet load (const std::string &root_file, const ImportOrderPolicy &);

  static DataSet blueprint_to_dray (const conduit::Node &n_dataset, const ImportOrderPolicy &);
};

} // namespace dray

#endif // DRAY_MFEM_READER_HPP

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BLUEPRINT_READER_HPP
#define DRAY_BLUEPRINT_READER_HPP

#include <conduit.hpp>
#include <dray/collection.hpp>

namespace dray
{

class BlueprintReader
{
  public:
  static Collection load (const std::string &root_file, const int cycle);

  static Collection load (const std::string &root_file);

  static DataSet blueprint_to_dray (const conduit::Node &n_dataset);
};

} // namespace dray

#endif // DRAY_MFEM_READER_HPP

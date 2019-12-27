// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MFEM_READER_HPP
#define DRAY_MFEM_READER_HPP

#include <dray/data_set.hpp>

namespace dray
{

class MFEMReader
{
  public:
  static DataSet
  load(const std::string &root_file, const int cycle = 0);
};

} // namespace dray

#endif // DRAY_MFEM_READER_HPP

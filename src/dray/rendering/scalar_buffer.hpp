// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SCALAR_BUFFER_HPP
#define DRAY_SCALAR_BUFFER_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

#include <conduit.hpp>
#include <vector>

namespace dray
{

struct ScalarBuffer
{
  std::vector<Array<float32>> m_scalars;
  std::vector<std::string>  m_names;
  Array<float32> m_depths;
  int32 m_width;
  int32 m_height;
  Float m_clear_value;

  void to_node(conduit::Node &mesh);
};

} // namespace dray
#endif

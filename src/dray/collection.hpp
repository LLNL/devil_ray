// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_COLLECTION_HPP
#define DRAY_COLLECTION_HPP

#include <dray/data_set.hpp>
#include <dray/types.hpp>

namespace dray
{

class Collection
{
protected:
  std::vector<DataSet> m_domains;
public:

  Collection();

  Range global_range(const std::string field_name);
  Range range(const std::string field_name);

  AABB<3> global_bounds();
  AABB<3> bounds();

  int32 topo_dims();
  int32 size();

  DataSet domain(int32 index);
};

} //namespace dray

#endif

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_NEW_DATA_SET_HPP
#define DRAY_NEW_DATA_SET_HPP

#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field_base.hpp>
#include <dray/topology_base.hpp>

#include <map>
#include <string>
#include <memory>

namespace dray
{

class nDataSet
{
protected:
  std::shared_ptr<TopologyBase> m_topo;
  std::vector<std::shared_ptr<FieldBase>> m_fields;
  bool m_is_valid;
public:
  nDataSet();
  nDataSet(std::shared_ptr<TopologyBase> topo);

  int32 number_of_fields() const;
  void topology(std::shared_ptr<TopologyBase> topo);
  bool has_field(const std::string &field_name) const;
  std::vector<std::string> fields() const;
  FieldBase* field(const std::string &field_name);
  TopologyBase* topology();
  void add_field(std::shared_ptr<FieldBase> field);
  friend class BlueprintReader;
};

} // namespace dray

#endif // DRAY_REF_POINT_HPP

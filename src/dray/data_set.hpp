// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DATA_SET_HPP
#define DRAY_DATA_SET_HPP

#include <dray/data_model/field.hpp>
#include <dray/data_model/mesh.hpp>
#include <conduit.hpp>

#include <map>
#include <string>
#include <memory>

namespace dray
{

class DataSet
{
protected:
  std::vector<std::shared_ptr<Mesh>> m_topos;
  std::vector<std::shared_ptr<Field>> m_fields;
  bool m_is_valid;
  int32 m_domain_id;
public:
  DataSet();
  DataSet(std::shared_ptr<Mesh> topo);

  void domain_id(const int32 id);
  int32 domain_id() const;


  void clear_topologies();
  int32 number_of_topologies() const;
  void add_topology(std::shared_ptr<Mesh> topo);
  bool has_topology(const std::string &topo_name) const;
  Mesh* topology(const int32 topo_index = 0);
  Mesh* topology(const std::string topo_name);
  std::vector<std::string> topologies() const;

  int32 number_of_fields() const;
  void clear_fields();
  bool has_field(const std::string &field_name) const;
  std::vector<std::string> fields() const;
  Field* field(const std::string &field_name);
  Field* field(const int &index);

  std::shared_ptr<Field> field_shared(const int &index);
  std::shared_ptr<Field> field_shared(const std::string &field_name);
  void add_field(std::shared_ptr<Field> field);
  friend class BlueprintReader;
  std::string field_info();

  // fill node with zero copied data
  void to_node(conduit::Node &n_dataset);

};

} // namespace dray

#endif // DRAY_REF_POINT_HPP

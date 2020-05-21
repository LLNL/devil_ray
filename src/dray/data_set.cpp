// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_set.hpp>
#include <dray/error.hpp>
#include <algorithm>
#include <sstream>

namespace dray
{

DataSet::DataSet(std::shared_ptr<TopologyBase> topo)
 : m_topo(topo),
   m_is_valid(true),
   m_domain_id(0)
{
}

DataSet::DataSet()
 : m_is_valid(false),
   m_domain_id(0)
{
}

void DataSet::domain_id(const int32 id)
{
  m_domain_id = id;
}

int32 DataSet::domain_id() const
{
  return m_domain_id;
}

void DataSet::topology(std::shared_ptr<TopologyBase> topo)
{
  m_topo = topo;
  m_is_valid = true;
  // should this invalidate the fields? Maybe we have a consistency check
}

int32 DataSet::number_of_fields() const
{
  return m_fields.size();
}

bool DataSet::has_field(const std::string &field_name) const
{
  bool found = false;

  for(int32 i = 0; i < m_fields.size(); ++i)
  {
    if(m_fields[i]->name() == field_name)
    {
      found = true;
      break;
    }
  }
  return found;
}

std::vector<std::string> DataSet::fields() const
{

  std::vector<std::string> names;
  for(int32 i = 0; i < m_fields.size(); ++i)
  {
    names.push_back(m_fields[i]->name());
  }
  return names;
}

FieldBase* DataSet::field(const int &index)
{
  if (index < 0 || index >= this->number_of_fields())
  {
    std::stringstream ss;
    ss<<"DataSet: Bad field index "<<index;
    DRAY_ERROR(ss.str());
  }
  return m_fields[index].get();
}

std::shared_ptr<FieldBase> DataSet::field_shared(const int &index)
{
  if (index < 0 || index >= this->number_of_fields())
  {
    std::stringstream ss;
    ss<<"DataSet: Bad field index "<<index;
    DRAY_ERROR(ss.str());
  }
  return m_fields[index];
}

FieldBase* DataSet::field(const std::string &field_name)
{
  bool found = false;
  int32 index = -1;
  for(int32 i = 0; i < m_fields.size(); ++i)
  {
    if(m_fields[i]->name() == field_name)
    {
      found = true;
      index = i;
      break;
    }
  }

  if (!found)
  {
    std::stringstream ss;
    ss << "Known fields: ";
    std::vector<std::string> names = this->fields();
    for (auto it = names.begin (); it != names.end (); ++it)
    {
      ss << "[" << *it << "] ";
    }

    DRAY_ERROR ("No field named '" + field_name + "' " + ss.str ());
  }

  return m_fields[index].get();
}

TopologyBase* DataSet::topology()
{
  if(!m_is_valid)
  {
    DRAY_ERROR ("Need to set the topology before asking for it.");
  }
  return m_topo.get();
}

void DataSet::add_field(std::shared_ptr<FieldBase> field)
{
  m_fields.push_back(field);
}

std::string DataSet::field_info()
{
  std::stringstream ss;
  for(int i = 0; i < m_fields.size(); ++i)
  {
    ss<<m_fields[i]->name()<<" "<<m_fields[i]->type_name()<<"\n";
  }
  return ss.str();
}

} // namespace dray

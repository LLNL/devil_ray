// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/new_data_set.hpp>
#include <dray/error.hpp>
#include <algorithm>
#include <sstream>

namespace dray
{

nDataSet::nDataSet(std::shared_ptr<TopologyBase> topo)
 : m_topo(topo),
   m_is_valid(true)
{
}

nDataSet::nDataSet()
 : m_is_valid(false)
{
}

void nDataSet::topology(std::shared_ptr<TopologyBase> topo)
{
  m_topo = topo;
  m_is_valid = true;
  // should this invalidate the fields? Maybe we have a consistency check
}

int32 nDataSet::number_of_fields() const
{
  return m_fields.size();
}

bool nDataSet::has_field(const std::string &field_name) const
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

std::vector<std::string> nDataSet::fields() const
{

  std::vector<std::string> names;
  for(int32 i = 0; i < m_fields.size(); ++i)
  {
    names.push_back(m_fields[i]->name());
  }
  return names;
}

FieldBase* nDataSet::field(const int &index)
{
  if (index < 0 || index >= this->number_of_fields())
  {
    std::stringstream ss;
    ss<<"DataSet: Bad field index "<<index;
    throw DRayError (ss.str());
  }
  return m_fields[index].get();
}

FieldBase* nDataSet::field(const std::string &field_name)
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

    throw DRayError ("No field named '" + field_name + "' " + ss.str ());
  }

  return m_fields[index].get();
}

TopologyBase* nDataSet::topology()
{
  if(!m_is_valid)
  {
    throw DRayError ("Need to set the topology before asking for it.");
  }
  return m_topo.get();
}

void nDataSet::add_field(std::shared_ptr<FieldBase> field)
{
  m_fields.push_back(field);
}

} // namespace dray

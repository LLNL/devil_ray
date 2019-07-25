#include <dray/data_set.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>

namespace dray
{

template<typename T>
DataSet<T>::DataSet(const Mesh<T> &mesh)
  : m_mesh(mesh),
    m_mesh_valid(true)
{
}

template<typename T>
DataSet<T>::DataSet()
  : m_mesh_valid(false)
{
}

template<typename T>
bool
DataSet<T>::has_field(const std::string &field_name)
{
  auto loc = m_field_names.find(field_name);
  bool res = false;
  if(loc != m_field_names.end())
  {
    res = true;
  }
  return res;
}

template<typename T>
void
DataSet<T>::add_field(const Field<T> &field, const std::string &field_name)
{
  if(has_field(field_name))
  {
    throw DRayError("Cannot add field '" + field_name + "'. Already exists");
  }

  m_fields.push_back(field);
  m_field_names.emplace(std::make_pair(field_name, m_fields.size() - 1));
}

template<typename T>
Field<T>
DataSet<T>::get_field(const std::string &field_name)
{
  if(!has_field(field_name))
  {
    throw DRayError("No field named '" + field_name +"'");
  }
  auto loc = m_field_names.find(field_name);
  return m_fields[loc->second];
}

template<typename T>
Field<T>
DataSet<T>::get_field(const int32 index)
{
  return m_fields.at(index);
}

template<typename T>
std::string
DataSet<T>::get_field_name(const int32 index)
{
  for (auto it = m_field_names.begin(); it != m_field_names.end(); ++it )
  {
    if (it->second == index)
    {
      return it->first;
    }
  }
  //TODO: do this better
  return "";
}

template<typename T>
int32
DataSet<T>::get_field_index(const std::string &field_name)
{
  int32 res = -1;
  auto loc = m_field_names.find(field_name);
  if(loc != m_field_names.end())
  {
    res = loc->second;
  }
  return res;
}

template<typename T>
int32
DataSet<T>::number_of_fields() const
{
  return m_fields.size();
}

template<typename T>
void
DataSet<T>::set_mesh(Mesh<T> &mesh)
{
  m_mesh_valid = true;
}

template<typename T>
Mesh<T>
DataSet<T>::get_mesh()
{
  if(!m_mesh_valid)
  {
    throw DRayError("DataSet: get_mesh called but no mesh was ever set");
  }
  return m_mesh;
}


// Explicit instantiations.
template class DataSet<float32>;
template class DataSet<float64>;
} // namespace dray

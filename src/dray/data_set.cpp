#include <dray/data_set.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>

namespace dray
{

template<typename T>
DataSet<T>::DataSet(const Mesh<T> &mesh)
  : m_mesh(mesh)
{
}

template<typename T>
bool
DataSet<T>::has_field(const std::string &field_name)
{
  auto loc = m_fields.find(field_name);
  bool res = false;
  if(loc != m_fields.end())
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

  m_fields.emplace(std::make_pair(field_name, field));
}

template<typename T>
Field<T>
DataSet<T>::get_field(const std::string &field_name)
{
  if(!has_field(field_name))
  {
    throw DRayError("No field named " + field_name);
  }
  auto loc = m_fields.find(field_name);
  return loc->second;
}

template<typename T>
Mesh<T>
DataSet<T>::get_mesh()
{
  return m_mesh;
}


// Explicit instantiations.
template class DataSet<float32>;
template class DataSet<float64>;
} // namespace dray

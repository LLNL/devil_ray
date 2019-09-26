#include <dray/data_set.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>

#include <sstream>

namespace dray
{
template<typename T, class ElemT>
DataSet<T, ElemT>::DataSet(const Mesh<T, ElemT> &mesh)
  : m_mesh(mesh),
    m_mesh_valid(true)
{
}

template<typename T, class ElemT>
DataSet<T, ElemT>::DataSet()
  : m_mesh_valid(false)
{
}


template <typename T, class ElemT>
std::set<std::string>
DataSet<T, ElemT>::list_fields()
{
  std::set<std::string> field_names;
  for (const auto & key__idx : m_field_names)
    field_names.emplace_hint(field_names.end(), key__idx.first);

  return field_names;
}

template<typename T, class ElemT>
bool
DataSet<T, ElemT>::has_field(const std::string &field_name)
{
  auto loc = m_field_names.find(field_name);
  bool res = false;
  if(loc != m_field_names.end())
  {
    res = true;
  }
  return res;
}

template<typename T, class ElemT>
void
DataSet<T, ElemT>::add_field(const Field<T, FieldOn<ElemT,1u>> &field, const std::string &field_name)
{
  if(has_field(field_name))
  {
    throw DRayError("Cannot add field '" + field_name + "'. Already exists");
  }

  m_fields.push_back(field);
  m_field_names.emplace(std::make_pair(field_name, m_fields.size() - 1));
}

template<typename T, class ElemT>
Field<T, FieldOn<ElemT,1u>>
DataSet<T, ElemT>::get_field(const std::string &field_name)
{
  if(!has_field(field_name))
  {
    std::stringstream ss;
    ss<<"Known fields: ";
    for(auto it = m_field_names.begin(); it != m_field_names.end(); ++it)
    {
      ss<<"["<<it->first<<"] ";
    }

    throw DRayError("No field named '" + field_name +"' " + ss.str());
  }
  auto loc = m_field_names.find(field_name);
  return m_fields[loc->second];
}

template<typename T, class ElemT>
Field<T, FieldOn<ElemT, 1u>>
DataSet<T, ElemT>::get_field(const int32 index)
{
  return m_fields.at(index);
}

template<typename T, class ElemT>
std::string
DataSet<T, ElemT>::get_field_name(const int32 index)
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

template<typename T, class ElemT>
int32
DataSet<T, ElemT>::get_field_index(const std::string &field_name)
{
  int32 res = -1;
  auto loc = m_field_names.find(field_name);
  if(loc != m_field_names.end())
  {
    res = loc->second;
  }
  return res;
}

template<typename T, class ElemT>
int32
DataSet<T, ElemT>::number_of_fields() const
{
  return m_fields.size();
}

template<typename T, class ElemT>
void
DataSet<T, ElemT>::set_mesh(Mesh<T, ElemT> &mesh)
{
  m_mesh = mesh;
  m_mesh_valid = true;
}

template<typename T, class ElemT>
Mesh<T, ElemT>
DataSet<T, ElemT>::get_mesh()
{
  if(!m_mesh_valid)
  {
    throw DRayError("DataSet: get_mesh called but no mesh was ever set");
  }
  return m_mesh;
}


// Explicit instantiations.
template class DataSet<float32, MeshElem<float32, 2u, ElemType::Quad, Order::General>>;
template class DataSet<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>>;
/// template class DataSet<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>;  // Can't activate triangle meshes until we fix ref_aabb-->SubRef<ElemT>
/// template class DataSet<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>;

template class DataSet<float64, MeshElem<float64, 2u, ElemType::Quad, Order::General>>;
template class DataSet<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>>;
/// template class DataSet<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>;
/// template class DataSet<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>;

} // namespace dray

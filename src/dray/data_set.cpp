#include <dray/data_set.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>

#include <sstream>

namespace dray
{
template<class ElemT>
DataSet<ElemT>::DataSet(const Mesh<ElemT> &mesh)
  : m_mesh(mesh),
    m_mesh_valid(true)
{
}

template<class ElemT>
DataSet<ElemT>::DataSet()
  : m_mesh_valid(false)
{
}


template <class ElemT>
std::set<std::string>
DataSet<ElemT>::fields()
{
  std::set<std::string> field_names;
  for (const auto & key__idx : m_field_names)
    field_names.emplace_hint(field_names.end(), key__idx.first);

  return field_names;
}

template<class ElemT>
bool
DataSet<ElemT>::has_field(const std::string &field_name)
{
  auto loc = m_field_names.find(field_name);
  bool res = false;
  if(loc != m_field_names.end())
  {
    res = true;
  }
  return res;
}

template<class ElemT>
void
DataSet<ElemT>::add_field(const Field<FieldOn<ElemT,1u>> &field, const std::string &field_name)
{
  if(has_field(field_name))
  {
    throw DRayError("Cannot add field '" + field_name + "'. Already exists");
  }

  m_fields.push_back(field);
  m_field_names.emplace(std::make_pair(field_name, m_fields.size() - 1));
}

template<class ElemT>
Field<FieldOn<ElemT,1u>>
DataSet<ElemT>::get_field(const std::string &field_name)
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

template<class ElemT>
Field<FieldOn<ElemT, 1u>>
DataSet<ElemT>::get_field(const int32 index)
{
  return m_fields.at(index);
}

template<class ElemT>
std::string
DataSet<ElemT>::get_field_name(const int32 index)
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

template<class ElemT>
int32
DataSet<ElemT>::get_field_index(const std::string &field_name)
{
  int32 res = -1;
  auto loc = m_field_names.find(field_name);
  if(loc != m_field_names.end())
  {
    res = loc->second;
  }
  return res;
}

template<class ElemT>
int32
DataSet<ElemT>::number_of_fields() const
{
  return m_fields.size();
}

template<class ElemT>
void
DataSet<ElemT>::set_mesh(Mesh<ElemT> &mesh)
{
  m_mesh = mesh;
  m_mesh_valid = true;
}

template<class ElemT>
Mesh<ElemT>
DataSet<ElemT>::get_mesh()
{
  if(!m_mesh_valid)
  {
    throw DRayError("DataSet: get_mesh called but no mesh was ever set");
  }
  return m_mesh;
}


// Explicit instantiations.
template class DataSet<MeshElem<2u, ElemType::Quad, Order::General>>;
template class DataSet<MeshElem<3u, ElemType::Quad, Order::General>>;
/// template class DataSet<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>;  // Can't activate triangle meshes until we fix ref_aabb-->SubRef<ElemT>
/// template class DataSet<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>;

/// template class DataSet<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>;
/// template class DataSet<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>;

} // namespace dray

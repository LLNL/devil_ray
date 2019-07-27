#include <dray/data_set.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>

namespace dray
{

template<typename T, class ElemT>
DataSet<T, ElemT>::DataSet(const Mesh<T, ElemT> &mesh)
  : m_mesh(mesh)
{
}

template<typename T, class ElemT>
bool
DataSet<T, ElemT>::has_field(const std::string &field_name)
{
  auto loc = m_fields.find(field_name);
  bool res = false;
  if(loc != m_fields.end())
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

  m_fields.emplace(std::make_pair(field_name, field));
}

template<typename T, class ElemT>
Field<T, FieldOn<ElemT,1u>>
DataSet<T, ElemT>::get_field(const std::string &field_name)
{
  if(!has_field(field_name))
  {
    throw DRayError("No field named " + field_name);
  }
  auto loc = m_fields.find(field_name);
  return loc->second;
}

template<typename T, class ElemT>
Mesh<T, ElemT>
DataSet<T, ElemT>::get_mesh()
{
  return m_mesh;
}


// Explicit instantiations.
template class DataSet<float32, MeshElem<float32, 2u, ElemType::Quad, Order::General>>;
template class DataSet<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>>;
template class DataSet<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>;
template class DataSet<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>;

template class DataSet<float64, MeshElem<float64, 2u, ElemType::Quad, Order::General>>;
template class DataSet<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>>;
template class DataSet<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>;
template class DataSet<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>;

} // namespace dray

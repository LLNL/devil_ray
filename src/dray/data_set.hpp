#ifndef DRAY_DATA_SET_HPP
#define DRAY_DATA_SET_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>

#include <map>
#include <set>
#include <string>

namespace dray
{

template<typename T, class ElemT>
class DataSet
{
protected:
  Mesh<T, ElemT> m_mesh;
  std::map<std::string,Field<T, FieldOn<ElemT, 1u>>> m_fields;
public:
  DataSet() = delete;
  DataSet(const Mesh<T, ElemT> &mesh);

  std::set<std::string> list_fields();

  void add_field(const Field<T, FieldOn<ElemT, 1u>> &field, const std::string &field_name);

  bool has_field(const std::string &field_name);

  Field<T, FieldOn<ElemT, 1u>> get_field(const std::string &field_name);

  Mesh<T, ElemT> get_mesh();

};

}

#endif//DRAY_REF_POINT_HPP

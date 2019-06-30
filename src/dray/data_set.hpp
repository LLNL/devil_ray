#ifndef DRAY_DATA_SET_HPP
#define DRAY_DATA_SET_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>

#include <map>
#include <string>

namespace dray
{

template<typename T>
class DataSet
{
protected:
  Mesh<T> m_mesh;
  std::map<std::string,Field<T>> m_fields;
public:
  DataSet() = delete;
  DataSet(const Mesh<T> &mesh);

  void add_field(const Field<T> &field, const std::string &field_name);

  bool has_field(const std::string &field_name);

  Field<T> get_field(const std::string &field_name);

  Mesh<T> get_mesh();

};

}

#endif//DRAY_REF_POINT_HPP

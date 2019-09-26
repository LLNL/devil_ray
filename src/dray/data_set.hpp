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
  std::vector<Field<T, FieldOn<ElemT, 1u>>> m_fields;
  std::map<std::string,int> m_field_names;
  bool m_mesh_valid;
public:
  DataSet();
  DataSet(const Mesh<T, ElemT> &mesh);

  std::set<std::string> list_fields();

  void add_field(const Field<T, FieldOn<ElemT, 1u>> &field, const std::string &field_name);

  bool has_field(const std::string &field_name);

  Field<T, FieldOn<ElemT, 1u>> get_field(const std::string &field_name);
  Field<T, FieldOn<ElemT, 1u>> get_field(const int32 index);
  std::string get_field_name(const int32 index);
  int32 get_field_index(const std::string &field_name);
  int32 number_of_fields() const;

  Mesh<T, ElemT> get_mesh();
  void set_mesh(Mesh<T, ElemT> &mesh);

};

}

#endif//DRAY_REF_POINT_HPP

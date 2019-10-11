#ifndef DRAY_DATA_SET_HPP
#define DRAY_DATA_SET_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>

#include <map>
#include <set>
#include <string>

namespace dray
{

template<class ElemT>
class DataSet
{
protected:
  Mesh<ElemT> m_mesh;
  std::vector<Field<FieldOn<ElemT, 1u>>> m_fields;
  std::map<std::string,int> m_field_names;
  bool m_mesh_valid;
public:
  DataSet();
  DataSet(const Mesh<ElemT> &mesh);

  std::set<std::string> list_fields();

  void add_field(const Field<FieldOn<ElemT, 1u>> &field, const std::string &field_name);

  bool has_field(const std::string &field_name);

  Field<FieldOn<ElemT, 1u>> get_field(const std::string &field_name);
  Field<FieldOn<ElemT, 1u>> get_field(const int32 index);
  std::string get_field_name(const int32 index);
  int32 get_field_index(const std::string &field_name);
  int32 number_of_fields() const;

  Mesh<ElemT> get_mesh();
  void set_mesh(Mesh<ElemT> &mesh);

};

}

#endif//DRAY_REF_POINT_HPP

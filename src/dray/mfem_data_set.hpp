#ifndef DRAY_MFEM_DATA_SET_HPP
#define DRAY_MFEM_DATA_SET_HPP

#include <dray/mfem_mesh.hpp>
#include <dray/mfem_grid_function.hpp>

#include <map>
#include <string>

namespace dray
{

class MFEMDataSet
{
protected:
  using FieldMap = std::map<std::string, MFEMGridFunction>;
  FieldMap  m_fields;
  MFEMMesh  m_mesh;


public:
  MFEMDataSet();
  MFEMDataSet(mfem::Mesh *mesh);
  MFEMDataSet(MFEMMesh mesh);

  //MFEMDataSet(const MFEMDataSet &other);

  void set_mesh(mfem::Mesh *mesh);
  MFEMMesh& get_mesh();
  
  void add_field(mfem::GridFunction *field, const std::string &name);
  bool has_field(const std::string &field_name);
  MFEMGridFunction& get_field(const std::string &field_name);
  int32 num_fields();

  void print_self();

};

} // namespace dray

#endif

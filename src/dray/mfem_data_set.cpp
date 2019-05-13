#include <dray/mfem_data_set.hpp>

#include <dray/error.hpp>

namespace dray
{

MFEMDataSet::MFEMDataSet()
{

}

//MFEMDataSet::MFEMDataSet(const MFEMDataSet &other)
//{
//}
//
MFEMDataSet::MFEMDataSet(mfem::Mesh *mesh)
 : m_mesh(mesh)
{

}

MFEMDataSet::MFEMDataSet(MFEMMesh mesh)
 : m_mesh(mesh)
{

}

void 
MFEMDataSet::set_mesh(mfem::Mesh *mesh)
{
  m_mesh.set_mesh(mesh);
}

MFEMMesh&
MFEMDataSet::get_mesh()
{
  return m_mesh;
}

void 
MFEMDataSet::add_field(mfem::GridFunction *field, const std::string &name)
{
  MFEMGridFunction gfield(field);
  m_fields[name] = gfield;
}

bool 
MFEMDataSet::has_field(const std::string &field_name)
{
  auto it = m_fields.find(field_name);
  return it != m_fields.end();
}

MFEMGridFunction&
MFEMDataSet::get_field(const std::string &field_name)
{
  if(!has_field(field_name))
  {
    std::string msg = "MFEMDataSet: no field named : " + field_name;
    throw DRayError(msg);
  }
  
  return m_fields[field_name];
}

int32 
MFEMDataSet::num_fields()
{
  return m_fields.size();
}

void 
MFEMDataSet::print_self()
{
  m_mesh.print_self();

  for(auto it = m_fields.begin(); it != m_fields.end(); ++it)
  {
    std::cout<<"Field: '"<<it->first<<"'\n";
    //it->second.print_self();
  }
}

} // namespace dray

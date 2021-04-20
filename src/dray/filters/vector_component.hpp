#ifndef DRAY_VECTOR_COMPONENT_HPP
#define DRAY_VECTOR_COMPONENT_HPP

#include <dray/collection.hpp>

namespace dray
{

class VectorComponent
{
protected:
  int32 m_component;
  std::string m_field_name;
  std::string m_output_name;
public:
  VectorComponent();
  void component(const int32 comp);
  void output_name(const std::string name);
  void field(const std::string name);
  Collection execute(Collection &collection);
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP

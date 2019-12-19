#ifndef DRAY_ISOSURFACE_HPP
#define DRAY_ISOSURFACE_HPP

#include <dray/new_data_set.hpp>
#include <dray/data_set.hpp>
#include <dray/color_table.hpp>
#include <dray/ray.hpp>
#include <dray/framebuffer.hpp>

namespace dray
{

class Isosurface
{
protected:
  std::string m_field_name;
  ColorTable m_color_table;
  float32 m_iso_value;
public:
  Isosurface();

  void execute(nDataSet &data_set,
               Array<Ray> &rays,
               Framebuffer &fb);

  template<class MeshElement, class FieldElement>
  void execute(Mesh<MeshElement> &mesh,
               Field<FieldElement> &field,
               Array<Ray> &rays,
               Framebuffer &fb);

  void set_field(const std::string field_name);
  void set_color_table(const ColorTable &color_table);
  void set_iso_value(const float32 iso_value);

};

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP

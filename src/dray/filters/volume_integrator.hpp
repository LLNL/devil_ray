#ifndef DRAY_VOLUME_INTEGRATOR_HPP
#define DRAY_VOLUME_INTEGRATOR_HPP

#include <dray/data_set.hpp>
#include <dray/color_table.hpp>
#include <dray/framebuffer.hpp>
#include <dray/ray.hpp>

namespace dray
{

class VolumeIntegrator
{
protected:
  std::string m_field_name;
  ColorTable m_color_table;
  int32 m_num_samples;

public:
  VolumeIntegrator();

  void execute(Array<Ray> &rays,
               DataSet &data_set,
               Framebuffer &fb);

  template<typename MeshElem, typename FieldElem>
  void trace(Mesh<MeshElem> &mesh,
             Field<FieldElem> &field,
             Array<Ray> &rays,
             Framebuffer &fb);

  void set_field(const std::string field_name);
  void set_color_table(const ColorTable &color_table);
  void set_num_samples(const int32 num_samples);

};

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP

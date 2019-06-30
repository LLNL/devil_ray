#ifndef DRAY_VOLUME_INTEGRATOR_HPP
#define DRAY_VOLUME_INTEGRATOR_HPP

#include <dray/data_set.hpp>
#include <dray/color_table.hpp>
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

  template<typename T>
  Array<Vec<float32,4>> execute(Array<Ray<T>> &rays,
                                DataSet<T> &data_set);

  void set_field(const std::string field_name);
  void set_color_table(const ColorTable &color_table);
  void set_num_samples(const int32 num_samples);

};

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP

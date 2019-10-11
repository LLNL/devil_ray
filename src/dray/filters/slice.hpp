#ifndef DRAY_SLICE_HPP
#define DRAY_SLICE_HPP

#include <dray/data_set.hpp>
#include <dray/color_table.hpp>
#include <dray/ray.hpp>

namespace dray
{

class Slice
{
protected:
  std::string m_field_name;
  ColorTable m_color_table;
  Vec<float32,3> m_point;
  Vec<float32,3> m_normal;
public:
  Slice();

  template<class ElemT>
  Array<Vec<float32,4>> execute(Array<Ray> &rays,
                                DataSet<ElemT> &data_set);

  void set_field(const std::string field_name);
  void set_color_table(const ColorTable &color_table);
  void set_point(const Vec<float32,3> &point);
  void set_normal(const Vec<float32,3> &normal);

};

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP

#ifndef DRAY_PATH_LENGTHS_HPP
#define DRAY_PATH_LENGTHS_HPP

#include <dray/data_set.hpp>
#include <dray/Element/elem_utils.hpp>

namespace dray
{

class PathLengths
{
protected:
  int32 m_x_res; // detector x resolution
  int32 m_y_res; // detector y resolution
  Float m_width; // detector width
  Float m_height;// detector height
  Vec<Float,3> m_point; // position of middle of the quad
  Vec<Float,3> m_normal; // quad orientation
  Vec<Float,3> m_x_dir;  // quad roll about the normal
  std::string m_absorption_field;
  std::string m_emission_field;
public:
  PathLengths();
  void execute(DataSet &data_set);
  Array<Vec<Float,3>> generate_pixels();
  void absorption_field(const std::string field_name);
  void emission_field(const std::string field_name);
  void write_image(Array<Float> values);
  void resolution(const int32 x, const int32 y);
  void size(const float32 width, const float32 height);
  void point(Vec<float32,3> p);
  //template<class ElemT>
  //DataSet execute(Mesh<ElemT> &mesh, DataSet &data_set);
};

};//namespace dray

#endif//DRAY_PATH_LENGTHS_HPP

#ifndef DRAY_FIRST_SCATTER_HPP
#define DRAY_FIRST_SCATTER_HPP

#include <dray/data_set.hpp>
#include <dray/collection.hpp>
#include <dray/Element/elem_utils.hpp>

namespace dray
{

class FirstScatter
{
protected:
  /// int32 m_x_res; // detector x resolution
  /// int32 m_y_res; // detector y resolution
  /// Float m_width; // detector width
  /// Float m_height;// detector height
  /// Vec<Float,3> m_point; // position of middle of the quad
  /// Vec<Float,3> m_normal; // quad orientation
  /// Vec<Float,3> m_x_dir;  // quad roll about the normal
  std::string m_total_cross_section_field;
  std::string m_emission_field;
  int32 m_legendre_order;

  // hack
  Float m_sigs;
public:
  FirstScatter();
  void execute(DataSet &data_set);
  void execute(Collection &collection);
  /// Array<Vec<Float,3>> generate_pixels();
  void total_cross_section_field(const std::string field_name);
  void emission_field(const std::string field_name);
  int32 legendre_order() const;
  void legendre_order(int32 l_order);

  // Hack. TODO import and use SigmaS matrix variable.
  void uniform_isotropic_scattering(Float sigs);

  /// void write_image(Array<Float> values);
  /// void resolution(const int32 x, const int32 y);
  /// void size(const float32 width, const float32 height);
  /// void point(Vec<float32,3> p);
  //template<class ElemT>
  //DataSet execute(Mesh<ElemT> &mesh, DataSet &data_set);
};

};//namespace dray

#endif//DRAY_FIRST_SCATTER_HPP

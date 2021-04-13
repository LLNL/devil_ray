#ifndef DRAY_FIRST_SCATTER_HPP
#define DRAY_FIRST_SCATTER_HPP

#include <dray/data_set.hpp>
#include <dray/collection.hpp>
#include <dray/Element/elem_utils.hpp>

namespace dray
{

class FirstScatter
{
public:
  enum ReturnType { ReturnUncollidedFlux, ReturnFirstScatter };

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
  std::string m_overwrite_first_scatter_field;
  int32 m_legendre_order;
  int32 m_face_quadrature_degree = 0;
  ReturnType m_ret;

  // hack
  Float m_sigs;
public:
  FirstScatter();
  void execute(DataSet &data_set);
  void execute(Collection &collection);

  // Absorption
  void total_cross_section_field(const std::string field_name);

  // Emission (original source)
  void emission_field(const std::string field_name);

  // Result of first scatter. Can be same as emission.
  void overwrite_first_scatter_field(const std::string field_name);

  int32 legendre_order() const;
  void legendre_order(int32 l_order);

  int32 face_quadrature_degree() const;
  void face_quadrature_degree(int32 degree);

  // Hack. TODO import and use SigmaS matrix variable.
  void uniform_isotropic_scattering(Float sigs);

  void return_type(ReturnType ret);
};

};//namespace dray

#endif//DRAY_FIRST_SCATTER_HPP

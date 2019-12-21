// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_CONTOUR_HPP
#define DRAY_CONTOUR_HPP

#include <dray/ray_tracing/traceable.hpp>

namespace dray
{
namespace ray_tracing
{

class Contour : public Traceable
{
protected:
  std::string m_field_name;
  float32 m_iso_value;
public:
  Contour() = delete;
  Contour(DataSet &data_set);

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays);

  template<class MeshElement, class FieldElement>
  Array<RayHit> execute(Mesh<MeshElement> &mesh,
                        Field<FieldElement> &field,
                        Array<Ray> &rays);

  void set_field(const std::string field_name);
  void set_iso_value(const float32 iso_value);

};

}};//namespace dray::ray_tracing

#endif//DRAY_CONTOUR_HPP

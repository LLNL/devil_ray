// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DERIVED_FIELD_HPP
#define DRAY_DERIVED_FIELD_HPP

#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/field_base.hpp>

namespace dray
{

//template<typename Element>
//class FField : public FieldBase
//{
//protected:
//  Field<Element> m_field;
//public:
//  FField() = delete;
//  FField(Field<Element> &field, const std::string name);
//  virtual ~FField();
//  virtual int32 order() const override;
//  Field<Element>& field();
//};
//
//// Element<topo dims, ncomps, base_shape, polynomial order>
//using Hex3  = Element<3u, 3u, ElemType::Quad, Order::General>;
//using Hex1  = Element<3u, 1u, ElemType::Quad, Order::General>;
//using Quad3 = Element<2u, 3u,ElemType::Quad, Order::General>;
//using Quad1 = Element<2u, 1u,ElemType::Quad, Order::General>;

} // namespace dray

#endif // DRAY_DERIVED_FIELD_HPP

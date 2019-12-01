// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/derived_field.hpp>
#include <string>

namespace dray
{

template<typename Element>
FField<Element>::FField(Field<Element> &field, const std::string name)
 : m_field(field)
{
  m_name = name;
}
template<typename Element>
FField<Element>::~FField()
{
}
template<typename Element>
int32 FField<Element>::order() const
{
  return m_field.get_poly_order();
}

template<typename Element>
Field<Element>& FField<Element>::field()
{
  return m_field;
}

// field types currently supported
template class FField<Quad3>;
template class FField<Quad1>;

template class FField<Hex3>;
template class FField<Hex1>;

} // namespace dray

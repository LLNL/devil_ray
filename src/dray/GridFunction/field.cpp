// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/GridFunction/field.hpp>
#include <dray/policies.hpp>

#include <dray/Element/element.hpp>

namespace dray
{

namespace detail
{
template <class ElemT> Range<> get_range (Field<ElemT> &field)
{
#warning "Need to do get_range by component"
  Range<> range;
  RAJA::ReduceMin<reduce_policy, Float> comp_min (infinity32 ());
  RAJA::ReduceMax<reduce_policy, Float> comp_max (neg_infinity32 ());

  const int32 num_nodes = field.get_dof_data ().m_values.size ();
  const Float *node_val_ptr =
  (const Float *)field.get_dof_data ().m_values.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_nodes), [=] DRAY_LAMBDA (int32 ii) {
    comp_min.min (node_val_ptr[ii]);
    comp_max.max (node_val_ptr[ii]);
  });

  range.include (comp_min.get ());
  range.include (comp_max.get ());
  return range;
}

} // namespace detail

template <class ElemT>
Field<ElemT>::Field (const GridFunctionData<ElemT::get_ncomp ()> &dof_data, int32 poly_order)
: m_dof_data (dof_data), m_poly_order (poly_order)
{
  m_range = detail::get_range (*this);
}

template <class ElemT> Range<> Field<ElemT>::get_range () const
{
  return m_range;
}
// Explicit instantiations.

template class FieldAccess<Element<2u, 1u, ElemType::Quad, Order::General>>;
template class FieldAccess<Element<2u, 3u, ElemType::Quad, Order::General>>;
template class FieldAccess<Element<2u, 1u, ElemType::Tri, Order::General>>;
template class FieldAccess<Element<2u, 3u, ElemType::Tri, Order::General>>;

template class FieldAccess<Element<3u, 1u, ElemType::Quad, Order::General>>;
template class FieldAccess<Element<3u, 3u, ElemType::Quad, Order::General>>;
template class FieldAccess<Element<3u, 1u, ElemType::Tri, Order::General>>;
template class FieldAccess<Element<3u, 3u, ElemType::Tri, Order::General>>;


// Explicit instantiations.
template class Field<Element<2u, 1u, ElemType::Quad, Order::General>>;
template class Field<Element<2u, 3u, ElemType::Quad, Order::General>>;
template class Field<Element<2u, 1u, ElemType::Tri, Order::General>>;
template class Field<Element<2u, 3u, ElemType::Tri, Order::General>>;

template class Field<Element<3u, 1u, ElemType::Quad, Order::General>>;
template class Field<Element<3u, 3u, ElemType::Quad, Order::General>>;
template class Field<Element<3u, 1u, ElemType::Tri, Order::General>>;
template class Field<Element<3u, 3u, ElemType::Tri, Order::General>>;


} // namespace dray

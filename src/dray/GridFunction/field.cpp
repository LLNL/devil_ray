// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/GridFunction/field.hpp>
#include <dray/policies.hpp>
#include <dray/error.hpp>

#include <dray/Element/element.hpp>


namespace dray
{

namespace detail
{
template <class ElemT> std::vector<Range> get_range (Field<ElemT> &field)
{

  RAJA::ReduceMin<reduce_policy, Float> comp_xmin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> comp_xmax (neg_infinity<Float>());

  RAJA::ReduceMin<reduce_policy, Float> comp_ymin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> comp_ymax (neg_infinity<Float>());

  RAJA::ReduceMin<reduce_policy, Float> comp_zmin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> comp_zmax (neg_infinity<Float>());

  const int32 num_nodes = field.get_dof_data ().m_values.size ();
  const int32 entries = num_nodes / ElemT::get_ncomp();

  constexpr int32 comps = ElemT::get_ncomp();
  assert(comps < 4);
  if(comps > 3)
  {
    DRAY_ERROR("We didn't plan for "<<comps<<" components");
  }

  const Vec<Float,comps> *node_val_ptr =
    field.get_dof_data ().m_values.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, entries), [=] DRAY_LAMBDA (int32 ii) {
    const Vec<Float,comps> value = node_val_ptr[ii];
    if(comps > 0)
    {
      comp_xmin.min (value[0]);
      comp_xmax.max (value[0]);
    }
    if(comps > 1)
    {
      comp_ymin.min (value[1]);
      comp_ymax.max (value[1]);
    }
    if(comps > 2)
    {
      comp_zmin.min (value[2]);
      comp_zmax.max (value[2]);
    }
  });

  std::vector<Range> ranges;
  if(comps > 0)
  {
    Range range;
    range.include (comp_xmin.get ());
    range.include (comp_xmax.get ());
    ranges.push_back(range);
  }
  if(comps > 1)
  {
    Range range;
    range.include (comp_ymin.get ());
    range.include (comp_ymax.get ());
    ranges.push_back(range);
  }
  if(comps > 2)
  {
    Range range;
    range.include (comp_zmin.get ());
    range.include (comp_zmax.get ());
    ranges.push_back(range);
  }
  return ranges;
}

} // namespace detail

template <class ElemT>
Field<ElemT>::Field (const GridFunction<ElemT::get_ncomp ()> &dof_data,
                     int32 poly_order,
                     const std::string name)
: m_dof_data (dof_data), m_poly_order (poly_order)
{
  m_ranges = detail::get_range (*this);
  this->name(name);
}

template <class ElemT> std::vector<Range> Field<ElemT>::range () const
{
  return m_ranges;
}

template <class ElemT>
int32 Field<ElemT>::order() const
{
  return m_poly_order;
}

template <class ElemT>
std::string Field<ElemT>::type_name() const
{
  return element_name<ElemT>(ElemT());
}


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

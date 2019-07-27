#include <dray/GridFunction/field.hpp>
#include <dray/policies.hpp>

#include <dray/Element/element.hpp>

namespace dray
{

namespace detail
{
template <typename T, class ElemT>
Range<> get_range(Field<T, ElemT> &field)
{
#warning "Need to do get_range by component"
  Range<> range;
  RAJA::ReduceMin<reduce_policy, T> comp_min(infinity32());
  RAJA::ReduceMax<reduce_policy, T> comp_max(neg_infinity32());

  const int32 num_nodes = field.get_dof_data().m_values.size();
  const T *node_val_ptr = (const T*) field.get_dof_data().m_values.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_nodes), [=] DRAY_LAMBDA (int32 ii)
  {
    comp_min.min(node_val_ptr[ii]);
    comp_max.max(node_val_ptr[ii]);
  });

  range.include( comp_min.get() );
  range.include( comp_max.get() );
  return range;
}

}

template <typename T, class ElemT>
Field<T, ElemT>::Field(const GridFunctionData<T, ElemT::get_ncomp()> &dof_data,
                               int32 poly_order)
  : m_dof_data(dof_data), m_poly_order(poly_order)
{
  m_range = detail::get_range(*this);
}

template <typename T, class ElemT>
Range<>
Field<T, ElemT>::get_range() const
{
  return m_range;
}
// Explicit instantiations.

template class FieldAccess<float32, Element<float32, 2u, 1u, ElemType::Quad, Order::General>>;
template class FieldAccess<float32, Element<float32, 2u, 3u, ElemType::Quad, Order::General>>;
template class FieldAccess<float32, Element<float32, 2u, 1u, ElemType::Tri, Order::General>>;
template class FieldAccess<float32, Element<float32, 2u, 3u, ElemType::Tri, Order::General>>;
template class FieldAccess<float64, Element<float64, 2u, 1u, ElemType::Quad, Order::General>>;
template class FieldAccess<float64, Element<float64, 2u, 3u, ElemType::Quad, Order::General>>;
template class FieldAccess<float64, Element<float64, 2u, 1u, ElemType::Tri, Order::General>>;
template class FieldAccess<float64, Element<float64, 2u, 3u, ElemType::Tri, Order::General>>;

template class FieldAccess<float32, Element<float32, 3u, 1u, ElemType::Quad, Order::General>>;
template class FieldAccess<float32, Element<float32, 3u, 3u, ElemType::Quad, Order::General>>;
template class FieldAccess<float32, Element<float32, 3u, 1u, ElemType::Tri, Order::General>>;
template class FieldAccess<float32, Element<float32, 3u, 3u, ElemType::Tri, Order::General>>;
template class FieldAccess<float64, Element<float64, 3u, 1u, ElemType::Quad, Order::General>>;
template class FieldAccess<float64, Element<float64, 3u, 3u, ElemType::Quad, Order::General>>;
template class FieldAccess<float64, Element<float64, 3u, 1u, ElemType::Tri, Order::General>>;
template class FieldAccess<float64, Element<float64, 3u, 3u, ElemType::Tri, Order::General>>;


// Explicit instantiations.
template class Field<float32, Element<float32, 2u, 1u, ElemType::Quad, Order::General>>;
template class Field<float32, Element<float32, 2u, 3u, ElemType::Quad, Order::General>>;
template class Field<float32, Element<float32, 2u, 1u, ElemType::Tri, Order::General>>;
template class Field<float32, Element<float32, 2u, 3u, ElemType::Tri, Order::General>>;
template class Field<float64, Element<float64, 2u, 1u, ElemType::Quad, Order::General>>;
template class Field<float64, Element<float64, 2u, 3u, ElemType::Quad, Order::General>>;
template class Field<float64, Element<float64, 2u, 1u, ElemType::Tri, Order::General>>;
template class Field<float64, Element<float64, 2u, 3u, ElemType::Tri, Order::General>>;

template class Field<float32, Element<float32, 3u, 1u, ElemType::Quad, Order::General>>;
template class Field<float32, Element<float32, 3u, 3u, ElemType::Quad, Order::General>>;
template class Field<float32, Element<float32, 3u, 1u, ElemType::Tri, Order::General>>;
template class Field<float32, Element<float32, 3u, 3u, ElemType::Tri, Order::General>>;
template class Field<float64, Element<float64, 3u, 1u, ElemType::Quad, Order::General>>;
template class Field<float64, Element<float64, 3u, 3u, ElemType::Quad, Order::General>>;
template class Field<float64, Element<float64, 3u, 1u, ElemType::Tri, Order::General>>;
template class Field<float64, Element<float64, 3u, 3u, ElemType::Tri, Order::General>>;


}

#include <dray/GridFunction/field.hpp>
#include <dray/policies.hpp>

namespace dray
{

namespace detail
{
template <typename T, int32 RefDim, int32 PhysDim>
Range<> get_range(Field<T, RefDim, PhysDim> &field)
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

template <typename T, int32 RefDim, int32 PhysDim>
Field<T,RefDim,PhysDim>::Field(const GridFunctionData<T,PhysDim> &dof_data,
                               int32 poly_order)
  : m_dof_data(dof_data), m_poly_order(poly_order)
{
  m_range = detail::get_range(*this);
}

template <typename T, int32 RefDim, int32 PhysDim>
Range<>
Field<T,RefDim,PhysDim>::get_range() const
{
  return m_range;
}
// Explicit instantiations.
template class FieldAccess<float32, 3,1>;     template class FieldAccess<float64, 3,1>;
//template class FieldAccess<float32, 3,2>;     template class FieldAccess<float64, 3,2>;
template class FieldAccess<float32, 3,3>;     template class FieldAccess<float64, 3,3>;



// Explicit instantiations.
template class Field<float32, 3,1>;     template class Field<float64, 3,1>;
//template class Field<float32, 3,2>;     template class Field<float64, 3,2>;
template class Field<float32, 3,3>;     template class Field<float64, 3,3>;

}

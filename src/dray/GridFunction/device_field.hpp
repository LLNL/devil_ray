// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_FIELD_HPP
#define DRAY_DEVICE_FIELD_HPP

#include <dray/Element/element.hpp>
#include <dray/GridFunction/grid_function.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/exports.hpp>
#include <dray/vec.hpp>

namespace dray
{
/*
 * @class FieldAccess
 * @brief Device-safe access to a collection of elements (just knows about the geometry, not fields).
 */
template <class ElemT> struct DeviceField
{
  static constexpr auto dim = ElemT::get_dim ();
  static constexpr auto ncomp = ElemT::get_ncomp ();
  static constexpr auto etype = ElemT::get_etype ();

  const int32 *m_idx_ptr;
  const Vec<Float, ncomp> *m_val_ptr;
  const int32 m_poly_order;

  DeviceField() = delete;
  DeviceField(Field<ElemT> &field);

  DRAY_EXEC ElemT get_elem (int32 el_idx) const;
};

template<class ElemT>
DeviceField<ElemT>::DeviceField(Field<ElemT> &field)
  : m_idx_ptr(field.m_dof_data.m_ctrl_idx.get_device_ptr_const()),
    m_val_ptr(field.m_dof_data.m_values.get_device_ptr_const()),
    m_poly_order(field.m_poly_order)
{
}

template <class ElemT>
DRAY_EXEC ElemT DeviceField<ElemT>::get_elem (int32 el_idx) const
{
  // We are just going to assume that the elements in the data store
  // are in the same position as their id, el_id==el_idx.
  ElemT ret;
  ReadDofPtr<Vec<Float, ncomp>> dof_ptr{ ElemT::get_num_dofs (m_poly_order) * el_idx + m_idx_ptr,
                                           m_val_ptr };
  ret.construct (el_idx, dof_ptr, m_poly_order);
  return ret;
}

} // namespace dray
#endif // DRAY_FIELD_HPP

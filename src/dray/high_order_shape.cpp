#include <dray/high_order_shape.hpp>
#include <dray/array_utils.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>

#include <dray/color_table.hpp>
#include <dray/point_location.hpp>

#include <assert.h>
#include <iostream>
#include <stdio.h>

namespace dray
{

//
// ElTransData::resize()
//
template <typename T, int32 PhysDim>
void 
ElTransData<T,PhysDim>::resize(int32 size_el, int32 el_dofs, int32 size_ctrl)
{
  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_ctrl;

  m_ctrl_idx.resize(size_el * el_dofs);
  m_values.resize(size_ctrl);
}


} // namespace dray

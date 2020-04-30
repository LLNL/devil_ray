// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/GridFunction/grid_function.hpp>
#include <dray/array_utils.hpp>

namespace dray
{

template <int32 PhysDim>
void GridFunction<PhysDim>::resize (int32 size_el, int32 el_dofs, int32 size_ctrl)
{
  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_ctrl;

  m_ctrl_idx.resize (size_el * el_dofs);
  m_values.resize (size_ctrl);
}

template <int32 PhysDim>
void GridFunction<PhysDim>::resize_counting (int32 size_el, int32 el_dofs)
{
  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_el * el_dofs;

  m_ctrl_idx = array_counting(size_el * el_dofs, 0, 1);
  m_values.resize (size_el * el_dofs);
}


template struct GridFunction<3>;
template struct GridFunction<1>;

} // namespace dray

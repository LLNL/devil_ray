// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/GridFunction/grid_function.hpp>
#include <dray/array_utils.hpp>

namespace dray
{

template <int32 PhysDim>
void GridFunction<PhysDim>::to_node(conduit::Node &n_gf)
{
  n_gf["dofs_per_element"] = m_el_dofs;
  n_gf["num_elemements"] = m_size_el;
  n_gf["values_size"] = m_values.size();
  n_gf["conn_size"] = m_size_ctrl;
  n_gf["phys_dim"] = PhysDim;

  Vec<Float,PhysDim> *values_ptr = m_values.get_host_ptr();
  Float *values_float_ptr = (Float*)(&(values_ptr[0][0]));
  n_gf["values"].set_external(values_float_ptr, m_values.size() * 3);

  int32 *conn_ptr = m_ctrl_idx.get_host_ptr();
  n_gf["conn"].set_external(conn_ptr, m_ctrl_idx.size());
}

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

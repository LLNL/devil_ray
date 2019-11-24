// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/el_trans.hpp>

namespace dray
{

//
// ElTransData::resize()
//
template <int32 PhysDim>
void ElTransData<PhysDim>::resize (int32 size_el, int32 el_dofs, int32 size_ctrl)
{
  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_ctrl;

  m_ctrl_idx.resize (size_el * el_dofs);
  m_values.resize (size_ctrl);
}

template struct ElTransData<3>;
template struct ElTransData<1>;

} // namespace dray

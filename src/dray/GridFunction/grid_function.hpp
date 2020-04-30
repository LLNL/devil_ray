// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_GRID_FUNCTION_DATA_HPP
#define DRAY_GRID_FUNCTION_DATA_HPP

#include <dray/array.hpp>
#include <dray/range.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <int32 PhysDim> struct GridFunction
{
  Array<int32> m_ctrl_idx; // 0 <= ii < size_el, 0 <= jj < el_dofs, 0 <= m_ctrl_idx[ii*el_dofs + jj] < size_ctrl
  Array<Vec<Float, PhysDim>> m_values; // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

  int32 m_el_dofs;
  int32 m_size_el;
  int32 m_size_ctrl;

  void resize (int32 size_el, int32 el_dofs, int32 size_ctrl);
  void resize_counting (int32 size_el, int32 el_dofs);

  int32 get_num_elem () const
  {
    return m_size_el;
  }

  template <typename CoeffIterType>
  DRAY_EXEC static void get_elt_node_range (const CoeffIterType &coeff_iter,
                                            const int32 el_dofs,
                                            Range *comp_range);
};

// TODO: I dont think this function belongs here. it doesnt'
// even access anything
template <int32 PhysDim>
template <typename CoeffIterType>
DRAY_EXEC void GridFunction<PhysDim>::get_elt_node_range (const CoeffIterType &coeff_iter,
                                                         const int32 el_dofs,
                                                         Range *comp_range)
{
  // Assume that each component range is already initialized.
  for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
  {
    Vec<Float, PhysDim> node_val = coeff_iter[dof_idx];
    for (int32 pdim = 0; pdim < PhysDim; pdim++)
    {
      comp_range[pdim].include (node_val[pdim]);
    }
  }
}

} // namespace dray
#endif // DRAY_GRID_FUNCTION_DATA_HPP

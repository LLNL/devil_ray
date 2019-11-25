// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_EL_TRANS_HPP
#define DRAY_EL_TRANS_HPP

#include <dray/array.hpp>
#include <dray/range.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <int32 PhysDim> struct ElTransData
{
  Array<int32> m_ctrl_idx; // 0 <= ii < size_el, 0 <= jj < el_dofs, 0 <= m_ctrl_idx[ii*el_dofs + jj] < size_ctrl
  Array<Vec<Float, PhysDim>> m_values; // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

  int32 m_el_dofs;
  int32 m_size_el;
  int32 m_size_ctrl;

  void resize (int32 size_el, int32 el_dofs, int32 size_ctrl);

  int32 get_num_elem () const
  {
    return m_size_el;
  }

  template <typename CoeffIterType>
  DRAY_EXEC static void get_elt_node_range (const CoeffIterType &coeff_iter,
                                            const int32 el_dofs,
                                            Range<> *comp_range);
};


//
// ElTransData::get_elt_node_range()
//
template <int32 PhysDim>
template <typename CoeffIterType>
DRAY_EXEC void ElTransData<PhysDim>::get_elt_node_range (const CoeffIterType &coeff_iter,
                                                         const int32 el_dofs,
                                                         Range<> *comp_range)
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


//
// ElTransPairOp  -- To superimpose a vector field and scalar field over the same reference space,
//                     without necessarily having the same numbers of degrees of freedom.
//
//                   Works best if the smaller dimension is Y.
template <class ElTransOpX, class ElTransOpY> struct ElTransPairOp
{
  static constexpr int32 phys_dim = ElTransOpX::phys_dim + ElTransOpY::phys_dim;
  static constexpr int32 ref_dim = ElTransOpX::ref_dim;
  static constexpr bool do_refs_agree = (ElTransOpX::ref_dim == ElTransOpY::ref_dim);

  DRAY_EXEC static bool is_inside (const Vec<Float, ref_dim> ref_pt)
  {
    return ElTransOpX::is_inside (ref_pt);
  }

  ElTransOpX trans_x;
  ElTransOpY trans_y;

  DRAY_EXEC void eval (const Vec<Float, ref_dim> &ref,
                       Vec<Float, phys_dim> &result_val,
                       Vec<Vec<Float, phys_dim>, ref_dim> &result_deriv)
  {
    constexpr int32 phys_x = ElTransOpX::phys_dim;
    constexpr int32 phys_y = ElTransOpY::phys_dim;

    Float *deriv_slots;

    trans_x.eval (ref, (Vec<Float, phys_x> &)result_val[0],
                  (Vec<Vec<Float, phys_x>, ref_dim> &)result_deriv[0]);

    // Shift X derivative values to the correct components.
    deriv_slots = (Float *)&result_deriv + phys_x * ref_dim - 1;
    for (int32 rdim = ref_dim - 1; rdim >= 0; rdim--)
      for (int32 pdim = phys_x - 1; pdim >= 0; pdim--, deriv_slots--)
        result_deriv[rdim][pdim] = *deriv_slots;

    Vec<Vec<Float, phys_y>, ref_dim> deriv_y;
    trans_y.eval (ref, (Vec<Float, phys_y> &)result_val[phys_x], deriv_y);

    // Copy Y derivative values to the correct components.
    deriv_slots = (Float *)&deriv_y;
    for (int32 rdim = 0; rdim < ref_dim; rdim++)
      for (int32 pdim = phys_x; pdim < phys_x + phys_y; pdim++, deriv_slots++)
        result_deriv[rdim][pdim] = *deriv_slots;
  }
};


//
// ElTransRayOp - Special purpose combination of element transformation and rays,
//                  PHI(u,v,...) - r(s),
//                where u,v,... are parametric space coordinates,
//                and s is distance along the ray.
//
//                Required: RayPhysDim <= ElTransOpType::phys_dim.
//
template <class ElTransOpType, int32 RayPhysDim>
struct ElTransRayOp : public ElTransOpType
{
  static constexpr int32 ref_dim = ElTransOpType::ref_dim + 1;
  static constexpr int32 phys_dim = ElTransOpType::phys_dim;

  Vec<Float, RayPhysDim> m_minus_ray_dir;

  DRAY_EXEC void set_minus_ray_dir (const Vec<Float, RayPhysDim> &ray_dir)
  {
    m_minus_ray_dir = -ray_dir;
  }

  DRAY_EXEC static bool is_inside (const Vec<Float, ref_dim> ref_pt)
  {
    return (ElTransOpType::is_inside ((const Vec<Float, ref_dim - 1> &)ref_pt) &&
            ref_pt[ref_dim - 1] > 0);
  }

  // Override eval().
  DRAY_EXEC void eval (const Vec<Float, ref_dim> &uvws,
                       Vec<Float, phys_dim> &result_val,
                       Vec<Vec<Float, phys_dim>, ref_dim> &result_deriv)
  {
    // Decompose uvws into disjoint reference coordinates.
    constexpr int32 uvw_dim = ElTransOpType::ref_dim;
    const Vec<Float, uvw_dim> &uvw = *((const Vec<Float, uvw_dim> *)&uvws);
    const Float &s = *((const Float *)&uvws[uvw_dim]);

    // Sub array of derivatives corresponding to uvw reference dimensions.
    Vec<Vec<Float, phys_dim>, uvw_dim> &uvw_deriv =
    *((Vec<Vec<Float, phys_dim>, uvw_dim> *)&result_deriv);

    ElTransOpType::eval (uvw, result_val, uvw_deriv);

    int32 pdim;
    for (pdim = 0; pdim < RayPhysDim; pdim++)
    {
      result_val[pdim] += m_minus_ray_dir[pdim] * s;
      result_deriv[uvw_dim][pdim] = m_minus_ray_dir[pdim];
    }
    for (; pdim < phys_dim; pdim++)
      result_deriv[uvw_dim][pdim] = 0;
  }
};

} // namespace dray

#endif

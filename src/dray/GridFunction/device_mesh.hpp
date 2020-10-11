// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_MESH_HPP
#define DRAY_DEVICE_MESH_HPP

#include <dray/Element/subref.hpp>
#include <dray/Element/element.hpp>
#include <dray/GridFunction/grid_function.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/aabb.hpp>
#include <dray/array_utils.hpp>
#include <dray/device_bvh.hpp>
#include <dray/exports.hpp>
#include <dray/location.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/vec.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

template <int32 dim, ElemType etype, int32 P_order>
using MeshElem = Element<dim, 3u, etype, P_order>;

/*
 * @class DeviceMesh
 * @brief Device-safe access to a collection of elements
 * (just knows about the geometry, not fields).
 */
template <class ElemT> struct DeviceMesh
{
  static constexpr auto dim = ElemT::get_dim ();
  static constexpr auto etype = ElemT::get_etype ();

  DeviceMesh (Mesh<ElemT> &mesh);
  DeviceMesh () = delete;

  //TODO use a DeviceGridFunction

  const int32 *m_idx_ptr;
  const Vec<Float, 3u> *m_val_ptr;
  const int32 m_poly_order;
  // bvh related data
  const DeviceBVH m_bvh;
  const SubRef<dim, etype> *m_ref_boxs;
  // if the element was subdivided m_ref_boxs
  // contains the sub-ref box of the original element
  // TODO: this should be married with BVH

  DRAY_EXEC_ONLY typename AdaptGetOrderPolicy<ElemT>::type get_order_policy() const
  {
    return adapt_get_order_policy(ElemT{}, m_poly_order);
  }

  DRAY_EXEC_ONLY ElemT get_elem (int32 el_idx) const;
  DRAY_EXEC_ONLY Location locate (const Vec<Float, 3> &point) const;
};


// ------------------ //
// DeviceMesh methods //
// ------------------ //

template <class ElemT>
DeviceMesh<ElemT>::DeviceMesh (Mesh<ElemT> &mesh)
: m_idx_ptr (mesh.m_dof_data.m_ctrl_idx.get_device_ptr_const ()),
  m_val_ptr (mesh.m_dof_data.m_values.get_device_ptr_const ()),
  m_poly_order (mesh.m_poly_order),
  m_bvh (mesh.get_bvh()),
  m_ref_boxs (mesh.m_ref_aabbs.get_device_ptr_const ())
{
}

template <class ElemT>
DRAY_EXEC_ONLY ElemT DeviceMesh<ElemT>::get_elem (int32 el_idx) const
{
  // We are just going to assume that the elements in the data store
  // are in the same position as their id, el_id==el_idx.
  ElemT ret;

  auto shape = adapt_get_shape(ElemT{});
  auto order_p = get_order_policy();
  const int32 dofs_per  = eattr::get_num_dofs(shape, order_p);

  const int32 elem_offset = dofs_per * el_idx;

  using DofVec = Vec<Float, 3u>;
  SharedDofPtr<DofVec> dof_ptr{ elem_offset + m_idx_ptr, m_val_ptr };
  ret.construct (el_idx, dof_ptr, m_poly_order);
  return ret;
}

//
// HACK to avoid calling eval_inverse() on 2x3 elements.
//
namespace detail
{
template <int32 d> struct LocateHack
{
};

// 3D: Works.
template <> struct LocateHack<3u>
{
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           stats::Stats &stats,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<3, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 3u> &ref_coords,
                                           bool use_init_guess = false)
  {
    /// return elem.eval_inverse (stats, world_coords, guess_domain, ref_coords, use_init_guess);

    // Bypass the subdivision search.
    //
    if (!use_init_guess)
      ref_coords = subref_center(guess_domain);
    return elem.eval_inverse_local (stats, world_coords, ref_coords);
  }

  // non-stats version
  template <class ElemT>
  static bool DRAY_EXEC_ONLY eval_inverse (const ElemT &elem,
                                           const Vec<typename ElemT::get_precision, 3u> &world_coords,
                                           const SubRef<3, ElemT::get_etype()> &guess_domain,
                                           Vec<typename ElemT::get_precision, 3u> &ref_coords,
                                           bool use_init_guess = false)
  {
    /// return elem.eval_inverse (world_coords, guess_domain, ref_coords, use_init_guess);

    // Bypass the subdivision search.
    if (!use_init_guess)
      ref_coords = subref_center(guess_domain);
    return elem.eval_inverse_local (world_coords, ref_coords);
  }
};
} // namespace detail

template <class ElemT>
DRAY_EXEC_ONLY Location DeviceMesh<ElemT>::locate (const Vec<Float, 3> &point) const
{
  constexpr auto etype = ElemT::get_etype ();  //TODO use type trait instead

  Location loc{ -1, { -1.f, -1.f, -1.f } };

  int32 todo[64];
  int32 current_node = 0;
  int32 stackptr = 0;

  constexpr int32 barrier = -2000000000;
  todo[stackptr] = barrier;
  while (current_node != barrier)
  {
    if (current_node > -1)
    {
      // inner node
      const Vec<float32, 4> first4 =
      const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 0]);
      const Vec<float32, 4> second4 =
      const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 1]);
      const Vec<float32, 4> third4 =
      const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 2]);

      bool in_left = true;
      if (point[0] < first4[0]) in_left = false;
      if (point[1] < first4[1]) in_left = false;
      if (point[2] < first4[2]) in_left = false;

      if (point[0] > first4[3]) in_left = false;
      if (point[1] > second4[0]) in_left = false;
      if (point[2] > second4[1]) in_left = false;

      bool in_right = true;
      if (point[0] < second4[2]) in_right = false;
      if (point[1] < second4[3]) in_right = false;
      if (point[2] < third4[0]) in_right = false;

      if (point[0] > third4[1]) in_right = false;
      if (point[1] > third4[2]) in_right = false;
      if (point[2] > third4[3]) in_right = false;

      if (!in_left && !in_right)
      {
        // pop the stack and continue
        current_node = todo[stackptr];
        stackptr--;
      }
      else
      {
        const Vec<float32, 4> children =
        const_get_vec4f (&m_bvh.m_inner_nodes[current_node + 3]);
        int32 l_child;
        constexpr int32 isize = sizeof (int32);
        // memcpy the int bits hidden in the floats
        memcpy (&l_child, &children[0], isize);
        int32 r_child;
        memcpy (&r_child, &children[1], isize);

        current_node = (in_left) ? l_child : r_child;

        if (in_left && in_right)
        {
          stackptr++;
          todo[stackptr] = r_child;
          // TODO: if we are in both children we could
          // go down the "closer" first by perhaps the distance
          // from the point to the center of the aabb
        }
      }
    }
    else
    {
      // leaf node
      // leafs are stored as negative numbers
      current_node = -current_node - 1; // swap the neg address
      const int32 el_idx = m_bvh.m_leaf_nodes[current_node];
      const int32 ref_box_id = m_bvh.m_aabb_ids[current_node];
      SubRef<dim, etype> ref_start_box = m_ref_boxs[ref_box_id];
      bool use_init_guess = true;
      // locate the point

      Vec<Float, dim> el_coords;

      bool found;
      found = detail::LocateHack<ElemT::get_dim ()>::template eval_inverse<ElemT> (
      get_elem (el_idx), point, ref_start_box, el_coords, use_init_guess);

      if (found)
      {
        loc.m_cell_id = el_idx;
        loc.m_ref_pt[0] = el_coords[0];
        loc.m_ref_pt[1] = el_coords[1];
        if (dim == 3)
        {
          loc.m_ref_pt[2] = el_coords[2];
        }
        break;
      }

      current_node = todo[stackptr];
      stackptr--;
    }
  } // while

  return loc;
}

} // namespace dray


#endif

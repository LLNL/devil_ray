#ifndef DRAY_DEVICE_MESH_HPP
#define DRAY_DEVICE_MESH_HPP

#include <dray/aabb.hpp>
#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/Element/element.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/location.hpp>
#include <dray/vec.hpp>
#include <dray/exports.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

template <uint32 dim, ElemType etype, Order P>
using MeshElem = Element<dim, 3u, etype, P>;

/*
 * @class DeviceMesh
 * @brief Device-safe access to a collection of elements
 * (just knows about the geometry, not fields).
 */
template <class ElemT>
struct DeviceMesh
{
  static constexpr auto dim = ElemT::get_dim();
  static constexpr auto etype = ElemT::get_etype();

  DeviceMesh(const Mesh<ElemT> &mesh);
  DeviceMesh() = delete;

  const int32 *m_idx_ptr;
  const Vec<Float,3u> *m_val_ptr;
  const int32 m_poly_order;
  //
  // get_elem()
  DRAY_EXEC ElemT get_elem(int32 el_idx) const;

};


// ------------------ //
// DeviceMesh methods //
// ------------------ //

template <class ElemT>
DeviceMesh<ElemT>::DeviceMesh(const Mesh<ElemT> &mesh)
  : m_idx_ptr(mesh.m_dof_data.m_ctrl_idx.get_device_ptr_const()),
    m_val_ptr(mesh.m_dof_data.m_values.get_device_ptr_const()),
    m_poly_order(mesh.m_poly_order)
{
}

template <class ElemT>
DRAY_EXEC ElemT
DeviceMesh<ElemT>::get_elem(int32 el_idx) const
{
  // We are just going to assume that the elements in the data store
  // are in the same position as their id, el_id==el_idx.
  ElemT ret;
  const int32 dofs_per = ElemT::get_num_dofs(m_poly_order);
  const int32 elem_offset = dofs_per * el_idx;

  using DofVec = Vec<Float,3u>;
  SharedDofPtr<DofVec> dof_ptr{elem_offset + m_idx_ptr, m_val_ptr};
  ret.construct(el_idx, dof_ptr, m_poly_order);
  return ret;
}

} // namespace dray


#endif

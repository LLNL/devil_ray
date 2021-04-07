// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <RAJA/RAJA.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>
#include <dray/aabb.hpp>
#include <dray/error_check.hpp>
#include <dray/array_utils.hpp>
#include <dray/dray.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/Element/element.hpp>


namespace dray
{

template <class ElemT> const BVH Mesh<ElemT>::get_bvh ()
{
  if(!m_is_constructed)
  {
    m_bvh = detail::construct_bvh (*this, m_ref_aabbs);
    m_is_constructed = true;
  }
  return m_bvh;
}

template <class ElemT>
Mesh<ElemT>::Mesh (const GridFunction<3u> &dof_data, int32 poly_order)
: m_dof_data (dof_data),
  m_poly_order (poly_order),
  m_is_constructed(false)
{
}

template <class ElemT>
Mesh<ElemT>::Mesh(const Mesh &other)
  : Mesh(other.m_dof_data,
         other.m_poly_order,
         other.m_is_constructed,
         other.m_bvh,
         other.m_ref_aabbs)
{
}

template <class ElemT>
Mesh<ElemT>::Mesh(Mesh &&other)
  : Mesh(other.m_dof_data,
         other.m_poly_order,
         other.m_is_constructed,
         other.m_bvh,
         other.m_ref_aabbs)
{
}

template <class ElemT> AABB<3> Mesh<ElemT>::get_bounds ()
{
  return get_bvh().m_bounds;
}

template <class ElemT>
Array<Location> Mesh<ElemT>::locate (Array<Vec<Float, 3u>> &wpoints)
{
  DRAY_LOG_OPEN ("locate");

  const int32 size = wpoints.size ();
  Array<Location> locations;
  locations.resize (size);

  Location *loc_ptr = locations.get_device_ptr ();
  const Vec<Float,3> *points_ptr = wpoints.get_device_ptr_const();

  DeviceMesh<ElemT> device_mesh (*this);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {

    Location loc = { -1, { -1.f, -1.f, -1.f } };
    const Vec<Float, 3> target_pt = points_ptr[i];
    loc_ptr[i] = device_mesh.locate(target_pt);
  });

  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();

  return locations;
}


// Explicit instantiations.
// template class MeshAccess<MeshElem<2u, ElemType::Tensor, Order::General>>;
// template class MeshAccess<MeshElem<2u, ElemType::Simplex, Order::General>>;
//
// template class MeshAccess<MeshElem<3u, ElemType::Tensor, Order::General>>;
// template class MeshAccess<MeshElem<3u, ElemType::Simplex, Order::General>>;

// Explicit instantiations.
template class Mesh<MeshElem<2u, ElemType::Tensor, Order::General>>;
template class Mesh<MeshElem<2u, ElemType::Tensor, Order::Linear>>;
template class Mesh<MeshElem<2u, ElemType::Tensor, Order::Quadratic>>;

template class Mesh<MeshElem<2u, ElemType::Simplex, Order::General>>;
template class Mesh<MeshElem<2u, ElemType::Simplex, Order::Linear>>;
template class Mesh<MeshElem<2u, ElemType::Simplex, Order::Quadratic>>;

template class Mesh<MeshElem<3u, ElemType::Tensor, Order::General>>;
template class Mesh<MeshElem<3u, ElemType::Tensor, Order::Linear>>;
template class Mesh<MeshElem<3u, ElemType::Tensor, Order::Quadratic>>;

template class Mesh<MeshElem<3u, ElemType::Simplex, Order::General>>;
template class Mesh<MeshElem<3u, ElemType::Simplex, Order::Linear>>;
template class Mesh<MeshElem<3u, ElemType::Simplex, Order::Quadratic>>;

} // namespace dray

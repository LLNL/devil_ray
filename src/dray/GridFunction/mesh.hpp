// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MESH_HPP
#define DRAY_MESH_HPP

#include <dray/Element/element.hpp>
#include <dray/GridFunction/grid_function.hpp>
#include <dray/aabb.hpp>
#include <dray/exports.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/location.hpp>
#include <dray/newton_solver.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/vec.hpp>
#include <dray/error.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

template <int32 dim, ElemType etype, int32 P_order>
using MeshElem = Element<dim, 3u, etype, P_order>;
// forward declare so we can have template friend
template <typename ElemT> class DeviceMesh;
template <typename ElemT> class Mesh;

/**
 * @class MeshFriend
 * @brief A mutual friend of all Mesh template class instantiations.
 * @note Avoids making all Mesh template instantiations friends of each other
 *       as well as friends to impostor 'Mesh' instantiations.
 */
class MeshFriend
{
  /**
   * @brief Use a fast path based on mesh order, or go back to general path.
   * @tparam new_order should equal the mesh order, or be -1.
   */
  public:
    template <class ElemT, int new_order>
    static
    Mesh<MeshElem<ElemT::get_dim(), ElemT::get_etype(), new_order>>
    to_fixed_order(Mesh<ElemT> &in_mesh);
};


/*
 * @class Mesh
 * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
 *
 * @warning Triangle and tet meshes are broken until we change ref aabbs to SubRef<etype>
 *          and implement reference space splitting for reference simplices.
 */
template <class ElemT> class Mesh
{
  public:
  static constexpr auto dim = ElemT::get_dim ();
  static constexpr auto etype = ElemT::get_etype ();

  protected:
  GridFunction<3u> m_dof_data;
  int32 m_poly_order;
  BVH m_bvh;
  Array<SubRef<dim, etype>> m_ref_aabbs;

  // Accept input data (as shared).
  // Useful for keeping same data but changing class template arguments.
  // If kept protected, can only be called by Mesh<ElemT> or friends of Mesh<ElemT>.
  Mesh(GridFunction<ElemT::get_ncomp()> dof_data,
       int32 poly_order,
       BVH bvh,
       Array<SubRef<dim, etype>> ref_aabbs)
    :
      m_dof_data{dof_data},
      m_poly_order{poly_order},
      m_bvh{bvh},
      m_ref_aabbs{ref_aabbs}
  { }


  public:
  friend class DeviceMesh<ElemT>;
  friend MeshFriend;

  Mesh () = delete;  // For now, probably need later.
  Mesh (const GridFunction<3u> &dof_data, int32 poly_order);
  // ndofs=3u because mesh always lives in 3D, even if it is a surface.

  Mesh(Mesh &&other);
  Mesh(const Mesh &other);

  /**
   * @brief Use a fast path based on mesh order, or go back to general path.
   * @tparam new_order should equal the mesh order, or be -1.
   */
  template <int new_order>
  Mesh<MeshElem<ElemT::get_dim(), ElemT::get_etype(), new_order>>
  to_fixed_order()
  {
    return MeshFriend::template to_fixed_order<ElemT, new_order>(*this);
  }


  int32 get_poly_order () const
  {
    return m_poly_order;
  }

  int32 get_num_elem () const
  {
    return m_dof_data.get_num_elem ();
  }

  const BVH get_bvh () const;

  AABB<3u> get_bounds () const;

  GridFunction<3u> get_dof_data ()
  {
    return m_dof_data;
  }

  const Array<SubRef<dim, etype>> &get_ref_aabbs () const
  {
    return m_ref_aabbs;
  }


  // Note: Do not use this for 2D meshes (TODO change interface so it is not possible to call)
  //       For now I have added a hack in the implementation that allows us to compile,
  //       but Mesh<2D>::locate() does not work at runtime.
  // TODO: matt note: i think locate should work in 2d since there are 2d simulations.
  //       Its essentially the same.
  //
  Array<Location> locate (Array<Vec<Float, 3>> &wpoints) const;

}; // Mesh



// MeshFriend::to_fixed_order()
//
//   Didn't want to write out all template argument combinations explicitly.
//   It's a template converter, should be ok as header.
//   Could go in a .tcc file.
//
template <class ElemT, int new_order>
Mesh<MeshElem<ElemT::get_dim(), ElemT::get_etype(), new_order>>
MeshFriend::to_fixed_order(Mesh<ElemT> &in_mesh)
{
  // Finite set of supported cases. Initially (bi/tri)quadrtic and (bi/tri)linear.
  static_assert(
      (new_order == -1 || new_order == 1 || new_order == 2),
      "Using fixed order 'new_order' not supported.\n"
      "Make sure Element<> for that order is instantiated "
      "and MeshFriend::to_fixed_order() "
      "is updated to include existing instantiations");

  if (!(new_order == -1 || new_order == in_mesh.get_poly_order()))
  {
    std::stringstream msg_ss;
    msg_ss << "Requested new_order (" << new_order
           << ") does not match existing poly order (" << in_mesh.get_poly_order()
           << ").";
    const std::string msg{msg_ss.str()};
    DRAY_ERROR(msg);
  }

  using NewElemT = MeshElem<ElemT::get_dim(), ElemT::get_etype(), new_order>;

  return Mesh<NewElemT>(in_mesh.m_dof_data,
                        in_mesh.m_poly_order,
                        in_mesh.m_bvh,
                        in_mesh.m_ref_aabbs);
}



} // namespace dray

#endif // DRAY_MESH_HPP

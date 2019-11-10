#ifndef DRAY_MESH_HPP
#define DRAY_MESH_HPP

#include <dray/aabb.hpp>
#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/Element/element.hpp>
#include <dray/newton_solver.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/location.hpp>
#include <dray/vec.hpp>
#include <dray/exports.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

template <uint32 dim, ElemType etype, Order P>
using MeshElem = Element<dim, 3u, etype, P>;
// forward declare so we can have template friend
template<typename ElemT> class DeviceMesh;

  /*
   * @class Mesh
   * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
   *
   * @warning Triangle and tet meshes are broken until we change ref aabbs to SubRef<etype>
   *          and implement reference space splitting for reference simplices.
   */
template <class ElemT>
class Mesh
{
public:
  static constexpr auto dim = ElemT::get_dim();
  static constexpr auto etype = ElemT::get_etype();
protected:
  GridFunctionData<3u> m_dof_data;
  int32 m_poly_order;
  BVH m_bvh;
  Array<AABB<dim>> m_ref_aabbs;
public:
  friend class DeviceMesh<ElemT>;

  Mesh();// = delete;  // For now, probably need later.
  Mesh(const GridFunctionData<3u> &dof_data, int32 poly_order);
  // ndofs=3u because mesh always lives in 3D, even if it is a surface.


  int32 get_poly_order() const { return m_poly_order; }

  int32 get_num_elem() const { return m_dof_data.get_num_elem(); }

  const BVH get_bvh() const;

  AABB<3u> get_bounds() const;

  GridFunctionData<3u> get_dof_data() { return m_dof_data; }

  const Array<AABB<dim>> & get_ref_aabbs() const { return m_ref_aabbs; }


  // Note: Do not use this for 2D meshes (TODO change interface so it is not possible to call)
  //       For now I have added a hack in the implementation that allows us to compile,
  //       but Mesh<2D>::locate() does not work at runtime.
  // TODO: matt note: i think locate should work in 2d since there are 2d simulations.
  //       Its essentially the same.
  //
  // TODO: remove active indices or come up with a better way to dispatch a subset of points
  //       Example: volume rendering compacting every X queries.
template <class StatsType>
void locate(Array<int32> &active_indices,
            Array<Vec<Float,3>> &wpoints,
            Array<Location> &locations,
            StatsType &stats) const;
#warning "remove active indices and come up with something better"
void locate(Array<int32> &active_indices,
            Array<Vec<Float,3>> &wpoints,
            Array<Location> &locations) const
{
#ifdef DRAY_STATS
  std::shared_ptr<stats::AppStats> app_stats_ptr =
    stats::global_app_stats.get_shared_ptr();
#else
  stats::NullAppStats n, *app_stats_ptr = &n;
#endif
  locate(active_indices, wpoints, locations, *app_stats_ptr);
}

}; // Mesh

} // namespace dray

#endif//DRAY_MESH_HPP

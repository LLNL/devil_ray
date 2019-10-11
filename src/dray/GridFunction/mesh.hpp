#ifndef DRAY_MESH_HPP
#define DRAY_MESH_HPP

#include <dray/aabb.hpp>
#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/Element/element.hpp>
#include <dray/newton_solver.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/ref_point.hpp>
#include <dray/vec.hpp>
#include <dray/exports.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

  template <uint32 dim, ElemType etype, Order P>
  using MeshElem = Element<dim, 3u, etype, P>;

  /*
   * @class MeshAccess
   * @brief Device-safe access to a collection of elements (just knows about the geometry, not fields).
   */
  template <class ElemT>
  struct MeshAccess
  {
    static constexpr auto dim = ElemT::get_dim();
    static constexpr auto etype = ElemT::get_etype();

    const int32 *m_idx_ptr;
    const Vec<Float,3u> *m_val_ptr;
    const int32  m_poly_order;

    //
    // get_elem()
    DRAY_EXEC ElemT get_elem(int32 el_idx) const;

    /// // world2ref()
    /// DRAY_EXEC bool
    /// world2ref(int32 el_idx,
    ///           const Vec<T,3u> &world_coords,
    ///           const SubRef<dim, etype> &guess_domain,
    ///           Vec<T,dim> &ref_coords,
    ///           bool use_init_guess = false) const;

    /// DRAY_EXEC bool
    /// world2ref(stats::IterativeProfile &iter_prof,
    ///           int32 el_idx,
    ///           const Vec<T,3u> &world_coords,
    ///           const SubRef<dim, etype> &guess_domain,
    ///           Vec<T,dim> &ref_coords,
    ///           bool use_init_guess = false) const;
  };


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

      Mesh();// = delete;  // For now, probably need later.
      Mesh(const GridFunctionData<3u> &dof_data, int32 poly_order);
      // ndofs=3u because mesh always lives in 3D, even if it is a surface.

      //
      // access_device_mesh() : Must call this BEFORE capture to RAJA lambda.
      MeshAccess<ElemT> access_device_mesh() const;

      //
      // access_host_mesh()
      MeshAccess<ElemT> access_host_mesh() const;

      //
      // get_poly_order()
      int32 get_poly_order() const { return m_poly_order; }

      //
      // get_num_elem()
      int32 get_num_elem() const { return m_dof_data.get_num_elem(); }

      const BVH get_bvh() const;

      AABB<3u> get_bounds() const;

      //
      // get_dof_data()  // TODO should this be removed?
      GridFunctionData<3u> get_dof_data() { return m_dof_data; }

      //
      // get_ref_aabbs()
      const Array<AABB<dim>> & get_ref_aabbs() const { return m_ref_aabbs; }


    //
    // locate()
    //
    // Note: Do not use this for 2D meshes (TODO change interface so it is not possible to call)
    //       For now I have added a hack in the implementation that allows us to compile,
    //       but Mesh<2D>::locate() does not work at runtime.
    //
    template <class StatsType>
    void locate(Array<int32> &active_indices,
                Array<Vec<Float,3>> &wpoints,
                Array<RefPoint<dim>> &rpoints,
                StatsType &stats) const;

    void locate(Array<int32> &active_indices,
                Array<Vec<Float,3>> &wpoints,
                Array<RefPoint<dim>> &rpoints) const
    {
#ifdef DRAY_STATS
      std::shared_ptr<stats::AppStats> app_stats_ptr =
        stats::global_app_stats.get_shared_ptr();
#else
      stats::NullAppStats n, *app_stats_ptr = &n;
#endif
      locate(active_indices, wpoints, rpoints, *app_stats_ptr);
    }
      protected:
        GridFunctionData<3u> m_dof_data;
        int32 m_poly_order;
        BVH m_bvh;
        Array<AABB<dim>> m_ref_aabbs;
    };

}

// Implementations (could go in a .tcc file and include that at the bottom of .hpp)

namespace dray
{

  // ------------------ //
  // MeshAccess methods //
  // ------------------ //

  //
  // get_elem()
  template <class ElemT>
  DRAY_EXEC ElemT
  MeshAccess<ElemT>::get_elem(int32 el_idx) const
  {
    // We are just going to assume that the elements in the data store
    // are in the same position as their id, el_id==el_idx.
    ElemT ret;
    SharedDofPtr<dray::Vec<Float,3u>> dof_ptr{ElemT::get_num_dofs(m_poly_order)*el_idx + m_idx_ptr, m_val_ptr};
    ret.construct(el_idx, dof_ptr, m_poly_order);
    return ret;
  }

  /// //
  /// // world2ref()
  /// template <typename T, class ElemT>
  /// DRAY_EXEC bool
  /// MeshAccess<T,ElemT>::world2ref(int32 el_idx,
  ///                              const Vec<T,3u> &world_coords,
  ///                              const SubRef<dim,etype> &guess_domain,
  ///                              Vec<T,dim> &ref_coords,
  ///                              bool use_init_guess) const


  /// {
  ///   return get_elem(el_idx).eval_inverse(world_coords,
  ///                                        guess_domain,
  ///                                        ref_coords,
  ///                                        use_init_guess);
  /// }
  /// template <typename T, class ElemT>
  /// DRAY_EXEC bool
  /// MeshAccess<T,ElemT>::world2ref(stats::IterativeProfile &iter_prof,
  ///                              int32 el_idx,
  ///                              const Vec<T,3u> &world_coords,
  ///                              const SubRef<dim,etype> &guess_domain,
  ///                              Vec<T,dim> &ref_coords,
  ///                              bool use_init_guess) const
  /// {
  ///   return get_elem(el_idx).eval_inverse(iter_prof,
  ///                                        world_coords,
  ///                                        guess_domain,
  ///                                        ref_coords,
  ///                                        use_init_guess);
  /// }


  // ---------------- //
  // Mesh methods     //
  // ---------------- //

  //
  // access_device_mesh()
  template <class ElemT>
  MeshAccess<ElemT> Mesh<ElemT>::access_device_mesh() const
  {
    return { m_dof_data.m_ctrl_idx.get_device_ptr_const(),
             m_dof_data.m_values.get_device_ptr_const(),
             m_poly_order };
  }

  //
  // access_host_mesh()
  template <class ElemT>
  MeshAccess<ElemT> Mesh<ElemT>::access_host_mesh() const
  {
    return { m_dof_data.m_ctrl_idx.get_host_ptr_const(),
             m_dof_data.m_values.get_host_ptr_const(),
             m_poly_order };
  }

} // namespace dray


#endif//DRAY_MESH_HPP

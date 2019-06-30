#ifndef DRAY_HIGH_ORDER_SHAPE_HPP
#define DRAY_HIGH_ORDER_SHAPE_HPP

#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>

#include <dray/array.hpp>
#include <dray/bernstein_basis.hpp>
#include <dray/el_trans.hpp>
#include <dray/math.hpp>
#include <dray/matrix.hpp>
#include <dray/ray.hpp>
#include <dray/ref_point.hpp>
#include <dray/types.hpp>

#include <dray/linear_bvh_builder.hpp>
#include <dray/aabb.hpp>
#include <dray/shading_context.hpp>

#include <dray/utils/appstats.hpp>

#include <stddef.h>
#include <string.h>  // memcpy()


namespace dray
{

struct IsoBVH : public BVH
{
  Range<> m_filter_range;

  IsoBVH() : BVH(), m_filter_range() {}
  IsoBVH(BVH &bvh, Range<> filter_range);
};

namespace detail
{
  template <typename T, int32 max_candidates>
  Array<int32> candidate_ray_intersection(Array<Ray<T>> rays, const BVH bvh);
}  // namespace detail


//
// MeshField
//
template <typename T>
class MeshField
{
public:
  static constexpr int32 ref_dim = 3;
  static constexpr int32 space_dim = 3; // Toplogical dims
  static constexpr int32 field_dim = 1;

  template <int32 _RefDim>
  using BShapeOp = BernsteinBasis<T,_RefDim>;

  using SpaceDataType = ElTransData<T,space_dim>;
  using FieldDataType = ElTransData<T,field_dim>;
  using SpaceTransOp  = ElTransOp<T, BernsteinBasis<T,ref_dim>, ElTransIter<T,space_dim>>;
  using FieldTransOp  = ElTransOp<T, BernsteinBasis<T,ref_dim>, ElTransIter<T,field_dim>>;


  MeshField(Mesh<T> mesh, Field<T> field) : m_mesh(mesh), m_field(field)
  {
    assert((mesh.get_num_elem() == field.get_num_elem()));

    m_size_el = mesh.get_num_elem();

    //Hack until we finish factoring out ElTrans stuff. TODO
    //m_eltrans_space = mesh.get_dof_data();
    //m_eltrans_field = field.get_dof_data();
    //m_p_space = mesh.get_poly_order();
    //m_p_field = field.get_poly_order();

    m_bvh = construct_bvh();
    m_iso_bvh.m_filter_range = Range<>();

    field_bounds(m_scalar_range);
  }

  // OLD. Use the other constructor, MeshField(Mesh,Field) if you can.
  //MeshField(SpaceDataType &eltrans_space,
  //          int32 poly_deg_space,
  //          FieldDataType &eltrans_field,
  //          int32 poly_deg_field)
  //  :
  //    m_mesh(eltrans_space, poly_deg_space),  // Works for now only because of the typedef in grid_function_data.hpp
  //    m_field(eltrans_field, poly_deg_field)
  //{
  //  assert(eltrans_space.m_size_el == eltrans_field.m_size_el);

  //  //TODO these will no longer exist once the other stuff stops depending on ElTransOp/ElTransData.
  //  m_eltrans_space = eltrans_space;
  //  m_eltrans_field = eltrans_field;
  //  m_p_space = poly_deg_space;
  //  m_p_field = poly_deg_field;
  //  m_size_el = eltrans_space.get_num_elem();

  //  m_bvh = construct_bvh();
  //  m_iso_bvh.m_filter_range = Range<>();

  //  field_bounds(m_scalar_range);
  //}
 ~MeshField() {}

  AABB<> get_bounds() const
  {
    return m_bvh.m_bounds;
  }

  Range<> get_scalar_range() const
  {
    return m_scalar_range;
  }

  // Store intersection into rays.
  template <class StatsType>
  void intersect_isosurface(Array<Ray<T>> rays, T isoval, Array<RefPoint<T,ref_dim>> &rpoints, StatsType &stats);

  void intersect_isosurface(Array<Ray<T>> rays, T isoval, Array<RefPoint<T,ref_dim>> &rpoints)
  {
#ifdef DRAY_STATS
    std::shared_ptr<stats::AppStats> app_stats_ptr = stats::global_app_stats.get_shared_ptr();
#else
    stats::NullAppStats n, *app_stats_ptr = &n;
#endif
    intersect_isosurface(rays, isoval, rpoints, *app_stats_ptr);
  }

  // Store intersection into rays.
  template <class StatsType>
  void intersect_mesh_boundary(Array<Ray<T>> rays, Array<RefPoint<T,ref_dim>> &rpoints, StatsType &stats);

  void intersect_mesh_boundary(Array<Ray<T>> rays, Array<RefPoint<T,ref_dim>> &rpoints)
  {
#ifdef DRAY_STATS
    std::shared_ptr<stats::AppStats> app_stats_ptr = stats::global_app_stats.get_shared_ptr();
#else
    stats::NullAppStats n, *app_stats_ptr = &n;
#endif
    intersect_mesh_boundary(rays, rpoints, *app_stats_ptr);
  }


  Array<ShadingContext<T>> get_shading_context(Array<Ray<T>> &rays, Array<RefPoint<T,ref_dim>> &rpoints) const;

  // Shade isosurface by gradient strength.
  Array<Vec<float32,4>> isosurface_gradient(Array<Ray<T>> rays, T isoval);

  // Helper functions. There should be no reason to use these outside the class.
  BVH construct_bvh();
  IsoBVH construct_iso_bvh(const Range<> &iso_range);
  void field_bounds(Range<> &scalar_range); // TODO move this capability into the bvh structure.

protected:
  BVH m_bvh;
  Range<> m_scalar_range;
  Mesh<T> m_mesh;
  Field<T> m_field;

  int32 m_size_el;
  IsoBVH m_iso_bvh;

  MeshField() = delete;  // Should never be called.
};

// Stub implementation   //TODO

template <typename T>
IsoBVH MeshField<T>::construct_iso_bvh(const Range<> &iso_range)
{
  //TODO  This method is supposed to filter out nodes from m_bvh that do not intersect iso_range.
  IsoBVH iso_bvh(m_bvh, iso_range);   // This uses m_bvh as is, and lies about the filter.
  return iso_bvh;
}




} // namespace dray

#endif



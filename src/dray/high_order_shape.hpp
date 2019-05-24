#ifndef DRAY_HIGH_ORDER_SHAPE_HPP
#define DRAY_HIGH_ORDER_SHAPE_HPP

#include <dray/array.hpp>
#include <dray/bernstein_basis.hpp>
#include <dray/el_trans.hpp>
#include <dray/math.hpp>
#include <dray/matrix.hpp>
#include <dray/ray.hpp>
#include <dray/types.hpp>

#include <dray/linear_bvh_builder.hpp>
#include <dray/aabb.hpp>
#include <dray/shading_context.hpp>

#include <stddef.h>
#include <string.h>  // memcpy()


namespace dray
{

struct IsoBVH : public BVH
{
  Range m_filter_range;

  IsoBVH() : BVH(), m_filter_range() {}
  IsoBVH(BVH &bvh, Range filter_range);
};

template<typename T>
struct DeviceFieldData
{
  const int32 m_el_dofs_space;
  const int32 m_el_dofs_field;

  const int32    *m_space_idx_ptr;
  const Vec<T,3> *m_space_val_ptr;
  const int32    *m_field_idx_ptr;
  const Vec<T,1> *m_field_val_ptr;

  const int32 m_p_space;
  const int32 m_p_field;
};

//
// MeshField
//
template <typename T>
class MeshField
{
public:
  static constexpr int32 ref_dim = 3;
  static constexpr int32 space_dim = 3;
  static constexpr int32 field_dim = 1;

  template <int32 _RefDim>
  using BShapeOp = BernsteinBasis<T,_RefDim>;

  using SpaceDataType = ElTransData<T,space_dim>;
  using FieldDataType = ElTransData<T,field_dim>;
  using SpaceTransOp  = ElTransOp<T, BernsteinBasis<T,ref_dim>, ElTransIter<T,space_dim>>;
  using FieldTransOp  = ElTransOp<T, BernsteinBasis<T,ref_dim>, ElTransIter<T,field_dim>>;

  MeshField(SpaceDataType &eltrans_space,
            int32 poly_deg_space,
            FieldDataType &eltrans_field,
            int32 poly_deg_field)
  {
    assert(eltrans_space.m_size_el == eltrans_field.m_size_el);

    m_eltrans_space = eltrans_space;
    m_eltrans_field = eltrans_field;
    m_p_space = poly_deg_space;
    m_p_field = poly_deg_field;
    m_size_el = eltrans_space.m_size_el;

    m_bvh = construct_bvh();
    m_iso_bvh.m_filter_range = Range();

    field_bounds(m_scalar_range);
  }
 ~MeshField() {}

  AABB get_bounds() const
  {
    return m_bvh.m_bounds;
  }

  Range get_scalar_range() const
  {
    return m_scalar_range;
  }

  void locate(Array<int32> &active_indices,
              Array<Ray<T>> &rays
#ifdef DRAY_STATS
              , Array<Ray<T>> &stat_rays
#endif
      ) const;

//  void locate(const Array<Vec<T,3>> points,
//              const Array<int32> active_idx,
//              Array<int32> &elt_ids,
//              Array<Vec<T,3>> &ref_pts
//#ifdef DRAY_STATS
//              , Array<Ray<T>> &stat_rays
//#endif
//      ) const;

  // Store intersection into rays.
  void intersect_isosurface(Array<Ray<T>> rays, T isoval);

  Array<ShadingContext<T>> get_shading_context(Array<Ray<T>> &rays) const;

  ////  // Volume integrator.
  Array<Vec<float32,4>> integrate(Array<Ray<T>> rays, T sample_dist) const;

  // Shade isosurface by gradient strength.
  Array<Vec<float32,4>> isosurface_gradient(Array<Ray<T>> rays, T isoval);

  // Helper functions. There should be no reason to use these outside the class.
  BVH construct_bvh();
  IsoBVH construct_iso_bvh(const Range &iso_range);
  void field_bounds(Range &scalar_range) const; // TODO move this capability into the bvh structure.
  DeviceFieldData<T> get_device_field_data() const;

  // Hack as we transition to decompose MeshField into Mesh and Field.
  SpaceDataType get_eltrans_data_space() const { return m_eltrans_space; }
  FieldDataType get_eltrans_data_field() const { return m_eltrans_field; }

protected:
  BVH m_bvh;
  Range m_scalar_range;
  SpaceDataType m_eltrans_space;
  FieldDataType m_eltrans_field;
  int32 m_p_space;
  int32 m_p_field;
  int32 m_size_el;
  IsoBVH m_iso_bvh;

  MeshField();  // Should never be called.
};

// Stub implementation   //TODO

template <typename T>
IsoBVH MeshField<T>::construct_iso_bvh(const Range &iso_range)
{
  //TODO  This method is supposed to filter out nodes from m_bvh that do not intersect iso_range.
  IsoBVH iso_bvh(m_bvh, iso_range);   // This uses m_bvh as is, and lies about the filter.
  return iso_bvh;
}




} // namespace dray

#endif



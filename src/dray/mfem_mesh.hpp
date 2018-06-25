#ifndef DRAY_MFEM_MESH_HPP
#define DRAY_MFEM_MESH_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/ray.hpp>
#include <mfem.hpp>

#include <dray/mfem_grid_function.hpp>

namespace dray
{

class MFEMMesh
{
protected:
  mfem::Mesh         *m_mesh;
  BVH                 m_bvh;
  bool                m_is_high_order;

  MFEMMesh(); 
public:
  MFEMMesh(mfem::Mesh *mesh);
  ~MFEMMesh(); 
  
  template<typename T>
  void            intersect(Ray<T> &rays);
  
    // Assumes that elt_ids and ref_pts have been sized to same length as points.
  template<typename T>
  void            locate(const Array<Vec<T,3>> points, Array<int32> &elt_ids, Array<Vec<T,3>> &ref_pts);

  template<typename T>
  void            locate(const Array<Vec<T,3>> points, const Array<int32> active_idx, Array<int32> &elt_ids, Array<Vec<T,3>> &ref_pts);

  AABB            get_bounds();

  void            print_self();
};


class MFEMMeshField : public MFEMMesh,
                      public MFEMGridFunction
{
public:
  MFEMMeshField(mfem::Mesh *mesh, mfem::GridFunction *gf);
  ~MFEMMeshField();

  // Use this method, advance rays, repeat.
  //
  // rays
  // isovalue
  // guesses_per_elt: The number of Newton solves to try per element per ray.
  template<typename T>
  void cast_to_isosurface(Ray<T> &rays, T isovalue, int32 guesses_per_elt);
};

} // namespace dray

#endif

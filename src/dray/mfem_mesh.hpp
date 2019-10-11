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

public:
  MFEMMesh();
  ~MFEMMesh();
  MFEMMesh(mfem::Mesh *mesh);

  void set_mesh(mfem::Mesh *mesh);

  void            intersect(Ray &rays);

    // Assumes that elt_ids and ref_pts have been sized to same length as points.
  void            locate(const Array<Vec<Float,3>> points, Array<Ray> &rays);

  void            locate(const Array<Vec<Float,3>> points,
                         const Array<int32> active_idx,
                         Array<Ray> &rays);

  AABB<>            get_bounds();

  void            print_self();
};

} // namespace dray

#endif

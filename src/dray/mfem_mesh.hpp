#ifndef DRAY_MFEM_MESH_HPP
#define DRAY_MFEM_MESH_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/ray.hpp>
#include <mfem.hpp>

namespace dray
{

class MFEMMesh
{
protected:
  mfem::Mesh     *m_mesh;
  AABB            m_bounds;
  BVH             m_bvh;
  bool            m_is_high_order;

  MFEMMesh(); 
public:
  MFEMMesh(mfem::Mesh *mesh); 
  ~MFEMMesh(); 
  
  template<typename T>
  void            intersect(Ray<T> &rays);

  AABB            get_bounds();

};

} // namespace dray

#endif

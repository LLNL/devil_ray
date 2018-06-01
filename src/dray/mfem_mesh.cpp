#include <dray/mfem_mesh.hpp>

namespace dray
{

namespace detail
{

} // namespace detail

MFEMMesh::MFEMMesh() 
{

}

MFEMMesh::MFEMMesh(mfem::Mesh *mesh)
{
  m_mesh = mesh;
}

MFEMMesh::~MFEMMesh()
{

}
  
template<typename T>
void
MFEMMesh::intersect(Ray<T> &rays)
{

}

AABB 
MFEMMesh::get_bounds()
{
  return m_bounds;
}

// explicit instantiations
template void MFEMMesh::intersect(ray32 &rays);
template void MFEMMesh::intersect(ray64 &rays);
} // namespace dray

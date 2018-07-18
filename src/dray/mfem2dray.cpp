#include <dray/mfem2dray.hpp>
#include <dray/types.hpp>

namespace dray
{

template <typename T>
ElTransData<T,3> import_mesh(const mfem::Mesh &mfem_mesh)
{
  const mfem::GridFunction *mesh_nodes;
  if ((mesh_nodes = mfem_mesh.GetNodes()) != NULL)
  {
    return import_grid_function_space<T>(*mesh_nodes);
  }
  else
  {
    return import_linear_mesh<T>(mfem_mesh);
  }
}

template <typename T>
ElTransData<T,3> import_linear_mesh(const mfem::Mesh &mfem_mesh)
{
  ElTransData<T,3> dataset;
  //TODO resize, import, etc.
  return dataset;
}

template <typename T>
ElTransData<T,3> import_grid_function_space(const mfem::GridFunction &mfem_gf)
{
  ElTransData<T,3> dataset;
  return dataset;
}

template <typename T>
ElTransData<T,1> import_grid_function_field(const mfem::GridFunction &mfem_gf)
{
  ElTransData<T,1> dataset;
  //TODO resize, import, etc.
  return dataset;
}


// Explicit instantiations
template ElTransData<float32,3> import_mesh<float32>(const mfem::Mesh &mfem_mesh);
template ElTransData<float32,3> import_linear_mesh<float32>(const mfem::Mesh &mfem_mesh);
template ElTransData<float32,3> import_grid_function_space<float32>(const mfem::GridFunction &mfem_gf);
template ElTransData<float32,1> import_grid_function_field<float32>(const mfem::GridFunction &mfem_gf);

template ElTransData<float64,3> import_mesh<float64>(const mfem::Mesh &mfem_mesh);
template ElTransData<float64,3> import_linear_mesh<float64>(const mfem::Mesh &mfem_mesh);
template ElTransData<float64,3> import_grid_function_space<float64>(const mfem::GridFunction &mfem_gf);
template ElTransData<float64,1> import_grid_function_field<float64>(const mfem::GridFunction &mfem_gf);

}  // namespace dray

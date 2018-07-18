#include <dray/mfem2dray.hpp>
#include <dray/types.hpp>

namespace dray
{

template <typename T>
HexETS<T> import_mesh(const mfem::Mesh &mfem_mesh)
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
HexETS<T> import_linear_mesh(const mfem::Mesh &mfem_mesh)
{
  HexETS<T> eltrans_space;

  //TODO

  return eltrans_space;
}


template <typename T>
HexETS<T> import_grid_function_space(const mfem::GridFunction &mfem_gf)
{
  HexETS<T> eltrans_space;

  //TODO     // I am here.

  return eltrans_space;
}

template <typename T>
HexETF<T> import_grid_function_field(const mfem::GridFunction &mfem_gf)
{
  HexETF<T> eltrans_field;

  // TODO

  return eltrans_field;
}


// Explicit instantiations
template HexETS<float32> import_mesh<float32>(const mfem::Mesh &mfem_mesh);
template HexETS<float32> import_linear_mesh<float32>(const mfem::Mesh &mfem_mesh);
template HexETS<float32> import_grid_function_space<float32>(const mfem::GridFunction &mfem_gf);
template HexETF<float32> import_grid_function_field<float32>(const mfem::GridFunction &mfem_gf);

template HexETS<float64> import_mesh<float64>(const mfem::Mesh &mfem_mesh);
template HexETS<float64> import_linear_mesh<float64>(const mfem::Mesh &mfem_mesh);
template HexETS<float64> import_grid_function_space<float64>(const mfem::GridFunction &mfem_gf);
template HexETF<float64> import_grid_function_field<float64>(const mfem::GridFunction &mfem_gf);

}  // namespace dray

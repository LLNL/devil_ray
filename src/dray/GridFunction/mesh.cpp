#include <dray/GridFunction/mesh.hpp>

namespace dray
{
 
  // Explicit instantiations.
  template class MeshAccess<float32, 3>;
  template class MeshAccess<float64, 3>;

  // Explicit instantiations.
  template class Mesh<float32, 3>;
  template class Mesh<float64, 3>;
}

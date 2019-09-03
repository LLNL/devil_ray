#include <dray/filters/mesh_boundary.hpp>

#include <dray/data_set.hpp>
#include <dray/GridFunction/mesh.hpp>


namespace dray
{

  template<typename T, class ElemT>
  DataSet<T, NDElem<ElemT, 2>> MeshBoundary::execute(DataSet<T, ElemT> &data_set)
  {
    using Elem3D = ElemT;
    using Elem2D = NDElem<ElemT, 2>;

    // Step 1: Extract the boundary mesh.

    GridFunctionData<T, 3u> mesh_data_2d;   // The 3u means 3 components (embedded in 3D).
    //TODO add mesh data.
    //use extract faces

    Mesh<T, Elem2D> boundary_mesh(mesh_data_2d, data_set.get_mesh().get_poly_order());
    DataSet<T, Elem2D> boundary_dataset(boundary_mesh);

    // Step 2: For each field, add boundary field to the boundary_dataset.
    //TODO

    return boundary_dataset;
  }



  // Explicit instantiations.
  //

  // <float32, Quad>
  template
    DataSet<float32, MeshElem<float32, 2u, ElemType::Quad, Order::General>>
    MeshBoundary::execute<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>>(
        DataSet<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>> &data_set);

  // <float32, Tri>
  /// template
  ///   DataSet<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>
  ///   MeshBoundary::execute<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>(
  ///       DataSet<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>> &data_set);

  // <float64, Quad>
  template
    DataSet<float64, MeshElem<float64, 2u, ElemType::Quad, Order::General>>
    MeshBoundary::execute<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>>(
        DataSet<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>> &data_set);

  // <float64, Tri>
  /// template
  ///   DataSet<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>
  ///   MeshBoundary::execute<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>(
  ///       DataSet<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>> &data_set);

}//namespace dray

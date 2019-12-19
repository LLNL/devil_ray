#ifndef DRAY_MESH_BOUNDARY_HPP
#define DRAY_MESH_BOUNDARY_HPP

#include <dray/new_data_set.hpp>
#include <dray/data_set.hpp>
#include <dray/Element/elem_utils.hpp>

namespace dray
{

class MeshBoundary
{
protected:
public:
  /**
   * @brief Extracts the boundary (surface) from a topologically 3D dataset,
   *        returing a topologically 2D dataset.
   * @param data_set The topologically 3D dataset.
   * @return A topologically 2D dataset. The Element types will be preserved
   *         except for the dimension.
   *
   * Assume that ElemT::get_dim()==3, so we will return NDElem with dim 2.
   */
  nDataSet execute(nDataSet &data_set);

  template<class ElemT>
  nDataSet execute(Mesh<ElemT> &mesh, nDataSet &data_set);
  //DataSet<NDElem<ElemT, 2>> execute(DataSet<ElemT> &data_set);
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP

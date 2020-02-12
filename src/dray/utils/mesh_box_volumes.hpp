// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MESH_BOX_VOLUMES_HPP
#define DRAY_MESH_BOX_VOLUMES_HPP

#include <dray/types.hpp>
#include <dray/exports.hpp>
#include <dray/array.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/data_set.hpp>

namespace dray
{
class MeshBoxVolumes
{
protected:
public:
  /**
   * @brief Returns the volume of the aabb bounds around each element in the mesh.
   */
  Array<Float> execute(DataSet &data_set);

  template<class ElemT>
  Array<Float> execute(Mesh<ElemT> &mesh, DataSet &data_set);
};

}//namespace dray

#endif//DRAY_MESH_BOX_VOLUMES_HPP

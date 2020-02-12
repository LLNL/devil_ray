// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/utils/mesh_box_volumes.hpp>

#include <dray/types.hpp>
#include <dray/data_set.hpp>
#include <dray/dispatcher.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/device_mesh.hpp>

namespace dray
{

struct MeshBoxVolFunctor
{
  MeshBoxVolumes *m_filter;
  DataSet m_input;
  Array<Float> m_output;

  MeshBoxVolFunctor(MeshBoxVolumes  *mesh_box_volumes,
                    DataSet &input)
    : m_filter(mesh_box_volumes),
      m_input(input)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    m_output = m_filter->execute(topo.mesh(), m_input);
  }

};


// execute()
Array<Float>
MeshBoxVolumes::execute(DataSet &data_set)
{
  MeshBoxVolFunctor func(this, data_set);
  dispatch_3d(data_set.topology(), func);
  return func.m_output;
}

// execute()
template <typename ElemT>
Array<Float>
MeshBoxVolumes::execute(Mesh<ElemT> &mesh, DataSet &data_set)
{
  DeviceMesh<ElemT> dmesh(mesh);
  Array<Float> volumes;
  const size_t num_elems = mesh.get_num_elem();
  volumes.resize(num_elems);
  Float * const volumes_ptr = volumes.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 eid) {
    AABB<ElemT::get_ncomp()> aabb;
    dmesh.get_elem(eid).get_bounds(aabb);
    volumes_ptr[eid] = aabb.area();
  });

  return volumes;
}






}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/filters/raise_degree.hpp>

#include <dray/data_set.hpp>
#include <dray/dispatcher.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/Element/element.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/array_utils.hpp>

#include <dray/policies.hpp>
#include <RAJA/RAJA.hpp>



namespace dray
{


struct RaiseDegDGFunctor
{
  RaiseDegreeDG *m_raise_deg;
  DataSet m_input;
  DataSet m_output;
  uint32 m_raise;

  RaiseDegDGFunctor(RaiseDegreeDG *raise_deg_filter,
          DataSet &input,
          uint32 raise)
    : m_raise_deg(raise_deg_filter),
      m_input(input),
      m_raise(raise)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    m_output = m_raise_deg->execute(topo.mesh(), m_input, m_raise);
  }
};

// execute()
DataSet
RaiseDegreeDG::execute(DataSet &data_set, uint32 raise)
{
  RaiseDegDGFunctor func(this, data_set, raise);
  dispatch_3d(data_set.topology(), func);
  return func.m_output;
}

// execute()
template <typename ElemT>
DataSet
RaiseDegreeDG::execute(Mesh<ElemT> &mesh, DataSet &data_set, uint32 raise)
{
  if (raise == 1)
    return this->template execute<ElemT, 1>(mesh, data_set);
  else if (raise == 2)
    return this->template execute<ElemT, 2>(mesh, data_set);
  else if (raise == 3)
    return this->template execute<ElemT, 3>(mesh, data_set);
  else
  {
    std::stringstream msg;
    msg<<"Raising degree by ("<<raise<<") at once is not supported.";
    DRAY_ERROR(msg.str());
    return data_set;
  }
}

// execute()
template <typename ElemT, uint32 raise>
DataSet
RaiseDegreeDG::execute(Mesh<ElemT> &mesh_lo, DataSet &data_set)
{
  //TODO in order to support fixed order ElemT, will need to
  //be able to convert fixed order into General order.

  const int32 p_lo = mesh_lo.get_poly_order();
  const int32 p_hi = p_lo + raise;
  const int32 npe_hi = ElemT::get_num_dofs(p_hi);

  const int32 num_elems = mesh_lo.get_num_elem();

  // Allocate L2 mesh, no sharing.
  GridFunction<3u> mesh_data_hi;
  mesh_data_hi.m_el_dofs = npe_hi;
  mesh_data_hi.m_size_el = num_elems;
  mesh_data_hi.m_size_ctrl = npe_hi * num_elems;
  mesh_data_hi.m_values.resize(npe_hi * num_elems);
  mesh_data_hi.m_ctrl_idx = array_counting(npe_hi * num_elems, 0, 1);

  // Projection.
  DeviceMesh<ElemT> dmesh_lo(mesh_lo);
  const int32 *ctrl_idx_ptr = mesh_data_hi.m_ctrl_idx.get_device_ptr_const();
  Vec<Float, 3u> *vals_ptr = mesh_data_hi.m_values.get_device_ptr();
  RAJA::forall<for_policy> (RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 eid) {
    WriteDofPtr<Vec<Float, 3u>> write_hi{ctrl_idx_ptr + eid * npe_hi, vals_ptr};

    ElemT elem_hi;
    elem_hi.construct(eid, write_hi.to_readonly_dof_ptr(), p_hi);

    ElemT::template project_to_higher_order_basis<raise>(dmesh_lo.get_elem(eid),
                                                         elem_hi,
                                                         write_hi);
  });
  //TODO if we do create a writeable element interface,
  //then use writeable device mesh above.

  // Below we will encapsulate the new mesh into a DataSet and return.
  // An alternative we could do is to steal the bvh of the new mesh
  // and give it to the old mesh.

  // GridFunction->Mesh->Topology->DataSet
  Mesh<ElemT> mesh_hi(mesh_data_hi, p_hi);
  ///DerivedTopology<ElemT> topo_hi(mesh_hi);
  DataSet data_set_hi(std::make_shared<DerivedTopology<ElemT>>(mesh_hi));
  for (const std::string &field_name : data_set.fields())
    data_set_hi.add_field(std::shared_ptr<FieldBase>(data_set.field(field_name)));

  return data_set_hi;
}


} //namespace dray

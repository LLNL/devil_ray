// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DISPATCHER_HPP
#define DRAY_DISPATCHER_HPP

#include<dray/derived_topology.hpp>
#include<dray/GridFunction/field.hpp>

namespace dray
{

// Scalar dispatch Design note: we can reduce this space since we already know
// the element type from the topology. That leaves only a couple
// possibilities: scalar and vector. This will get more complicated
// when we open up the paths for order specializations.
// I feel dirty.

// Figure out a way to specialize based on TopoType
// No need to even call hex when its a quad topo
template<typename Topology, typename Functor>
void dispatch_scalar_field(FieldBase *field, Topology *topo, Functor &func)
{
  using MeshElement = typename Topology::ElementType;

  using ScalarElement = Element<MeshElement::get_dim(),
                        1, // one component
                        MeshElement::get_etype(),
                        MeshElement::get_P ()>;

  if(dynamic_cast<Field<ScalarElement>*>(field) != nullptr)
  {
    Field<ScalarElement>* scalar_field  = dynamic_cast<Field<ScalarElement>*>(field);
    func(*topo, *scalar_field);
    std::cout<<"scalar field 1\n";
  }
  else
  {
    std::cout<<"field cast failed\n";
  }
}

template<typename Functor>
void dispatch_3d(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if(dynamic_cast<HexTopology*>(topo) != nullptr)
  {
    HexTopology *hex_topo = dynamic_cast<HexTopology*>(topo);
    std::cout<<"hex 1\n";
    dispatch_scalar_field(field, hex_topo, func);
  }
  else
  {
    // we don't support this type
  }
}

template<typename Functor>
void dispatch_3d_topology(TopologyBase *topo, Functor &func)
{
  if(dynamic_cast<HexTopology*>(topo) != nullptr)
  {
    HexTopology *hex_topo = dynamic_cast<HexTopology*>(topo);
    std::cout<<"3d topo hex\n";
    func(*hex_topo);
  }
  else
  {
    // we don't support this type
  }
}

// Topologically 2d
template<typename Functor>
void dispatch_2d(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if(dynamic_cast<QuadTopology*>(topo) != nullptr)
  {
    QuadTopology *quad_topo = dynamic_cast<QuadTopology*>(topo);
    std::cout<<"quad 1\n";
    dispatch_scalar_field(field, quad_topo, func);
  }
  else
  {
    // we don't support this type
  }
}

} // namespace dray
#endif

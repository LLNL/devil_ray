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

template<typename Topology, typename Functor>
void dispatch_scalar_field(FieldBase *field, Topology *topo, Functor &func)
{
  if(dynamic_cast<Field<Hex1>*>(field) != nullptr)
  {
    Field<Hex1>* hex1 = dynamic_cast<Field<Hex1>*>(field);
    func(*topo, *hex1);
    std::cout<<"hex field 1\n";
  }
  else if(dynamic_cast<Quad1*>(field) != nullptr)
  {
    std::cout<<"quad field 1\n";
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

} // namespace dray
#endif

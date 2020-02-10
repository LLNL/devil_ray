// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DISPATCHER_HPP
#define DRAY_DISPATCHER_HPP

#include<dray/derived_topology.hpp>
#include<dray/GridFunction/field.hpp>
#include<dray/error.hpp>

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
  }
  else
  {
    std::stringstream msg;
    msg<<"Cast of field '"<<field->type_name()<<"' failed ";
    msg<<"("<<__FILE__<<", "<<__LINE__<<")\n";
    DRAY_ERROR(msg.str());
  }
}

template<typename Functor>
void dispatch_3d(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if(dynamic_cast<HexTopology*>(topo) != nullptr)
  {
    HexTopology *hex_topo = dynamic_cast<HexTopology*>(topo);
    dispatch_scalar_field(field, hex_topo, func);
  }
  else
  {
    std::stringstream msg;
    msg<<"Cast of topology '"<<topo->type_name()<<"' failed ";
    msg<<"("<<__FILE__<<", "<<__LINE__<<")\n";
    DRAY_ERROR(msg.str());
  }
}

template<typename Functor>
void dispatch(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if(dynamic_cast<HexTopology*>(topo) != nullptr)
  {
    HexTopology *hex_topo = dynamic_cast<HexTopology*>(topo);
    dispatch_scalar_field(field, hex_topo, func);
  }
  else if(dynamic_cast<QuadTopology*>(topo) != nullptr)
  {
    QuadTopology *hex_topo = dynamic_cast<QuadTopology*>(topo);
    dispatch_scalar_field(field, hex_topo, func);
  }
  else
  {
    std::stringstream msg;
    msg<<"Cast of topology '"<<topo->type_name()<<"' failed ";
    msg<<"("<<__FILE__<<", "<<__LINE__<<")\n";
    DRAY_ERROR(msg.str());
  }
}

template<typename Functor>
void dispatch_3d(TopologyBase *topo, Functor &func)
{
  if(dynamic_cast<HexTopology*>(topo) != nullptr)
  {
    HexTopology *hex_topo = dynamic_cast<HexTopology*>(topo);
    func(*hex_topo);
  }
  else
  {
    std::stringstream msg;
    msg<<"Cast of topology '"<<topo->type_name()<<"' failed.";
    DRAY_ERROR(msg.str());
  }
}

template<typename Functor>
void dispatch_3d(FieldBase *field, Functor &func)
{
  if(dynamic_cast<Field<HexScalar>*>(field) != nullptr)
  {
    Field<HexScalar>* scalar_field  = dynamic_cast<Field<HexScalar>*>(field);
    func(*scalar_field);
  }
  else
  {
    std::stringstream msg;
    msg<<"Cast of topology '"<<field->type_name()<<"' failed.";
    DRAY_ERROR(msg.str());
  }
}

// Topologically 2d
template<typename Functor>
void dispatch_2d(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if(dynamic_cast<QuadTopology*>(topo) != nullptr)
  {
    QuadTopology *quad_topo = dynamic_cast<QuadTopology*>(topo);
    dispatch_scalar_field(field, quad_topo, func);
  }
  else
  {
    // we don't support this type
    std::stringstream msg;
    msg<<"Cast of topology '"<<topo->type_name()<<"' failed.";
    DRAY_ERROR(msg.str());
  }
}

template<typename Functor>
void dispatch_2d(TopologyBase *topo, Functor &func)
{
  if(dynamic_cast<QuadTopology*>(topo) != nullptr)
  {
    QuadTopology *quad_topo = dynamic_cast<QuadTopology*>(topo);
    func(*quad_topo);
  }
  else
  {
    std::stringstream msg;
    msg<<"Cast of topology '"<<topo->type_name()<<"' failed.";
    DRAY_ERROR(msg.str());
  }
}

} // namespace dray
#endif

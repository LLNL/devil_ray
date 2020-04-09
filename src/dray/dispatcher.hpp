// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DISPATCHER_HPP
#define DRAY_DISPATCHER_HPP

#include<dray/topology_base.hpp>
#include<dray/derived_topology.hpp>
#include<dray/GridFunction/field.hpp>
#include<dray/error.hpp>
#include<dray/utils/data_logger.hpp>

#include <utility>
#include <type_traits>

namespace dray
{

namespace detail
{
  void cast_topo_failed(TopologyBase *topo, const char *file, unsigned long long line);
  void cast_field_failed(FieldBase *field, const char *file, unsigned long long line);
}

// Scalar dispatch Design note: we can reduce this space since we already know
// the element type from the topology. That leaves only a couple
// possibilities: scalar and vector. This will get more complicated
// when we open up the paths for order specializations.
// I feel dirty.

// Figure out a way to specialize based on TopoType
// No need to even call hex when its a quad topo
template<typename DerivedTopologyT, typename Functor>
void dispatch_scalar_field(FieldBase *field, DerivedTopologyT *topo, Functor &func)
{
  using MeshElement = typename DerivedTopologyT::ElementType;

  using ScalarElement = Element<MeshElement::get_dim(),
                        1, // one component
                        MeshElement::get_etype(),
                        Order::General>;
  //TODO Do not assume the scalar order policy is Order::General.
  //  After create Field::to_fixed_order() and instantiations,
  //  upgrade this dispatcher.

  if(dynamic_cast<Field<ScalarElement>*>(field) != nullptr)
  {
    Field<ScalarElement>* scalar_field  = dynamic_cast<Field<ScalarElement>*>(field);
    func(*topo, *scalar_field);
  }
  else
    detail::cast_field_failed(field, __FILE__, __LINE__);
}


// ======================================================================
// Helpers
//   dispatch_topo_field()
//   dispatch_topo_only()
//   dispatch_field_only()
// ======================================================================

template <typename TopologyGuessT, typename Functor>
bool dispatch_topo_field(TopologyGuessT *, TopologyBase *topo, FieldBase *field, Functor &func)
{
  static_assert(!std::is_same<const TopologyGuessT*, const TopologyBase*>::value,
      "Cannot dispatch to TopologyBase. (Did you mix up tag and pointer?)");

  TopologyGuessT *derived_topo;

  if ((derived_topo = dynamic_cast<TopologyGuessT*>(topo)) != nullptr)
  {
    DRAY_INFO("Dispatched " + topo->type_name() + " topology to " + element_name<typename TopologyGuessT::ElementType>());
    dispatch_scalar_field(field, derived_topo, func);
  }

  return (derived_topo != nullptr);
}

template <typename TopologyGuessT, typename Functor>
bool dispatch_topo_only(TopologyGuessT *, TopologyBase *topo, Functor &func)
{
  static_assert(!std::is_same<const TopologyGuessT*, const TopologyBase*>::value,
      "Cannot dispatch to TopologyBase. (Did you mix up tag and pointer?)");

  TopologyGuessT *derived_topo;

  if ((derived_topo = dynamic_cast<TopologyGuessT*>(topo)) != nullptr)
  {
    DRAY_INFO("Dispatched " + topo->type_name() + " topology to " + element_name<typename TopologyGuessT::ElementType>());
    func(*derived_topo);
  }

  return (derived_topo != nullptr);
}

// ======================================================================

//
// Dispatch with (topo, field, func).
//
template<typename Functor>
bool dispatch_3d(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if (!dispatch_topo_field((HexTopology*)0,    topo, field, func) &&
      !dispatch_topo_field((HexTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field((HexTopology_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TetTopology*)0,    topo, field, func) &&
      !dispatch_topo_field((TetTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field((TetTopology_P2*)0, topo, field, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
  return true;
}

// Topologically 2d
template<typename Functor>
bool dispatch_2d(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if (!dispatch_topo_field((QuadTopology*)0,    topo, field, func) &&
      !dispatch_topo_field((QuadTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field((QuadTopology_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TriTopology*)0,     topo, field, func) &&
      !dispatch_topo_field((TriTopology_P1*)0,  topo, field, func) &&
      !dispatch_topo_field((TriTopology_P2*)0,  topo, field, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
  return true;
}

template<typename Functor>
void dispatch(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if (!dispatch_topo_field((HexTopology*)0,    topo, field, func) &&
      !dispatch_topo_field((HexTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field((HexTopology_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TetTopology*)0,    topo, field, func) &&
      !dispatch_topo_field((TetTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field((TetTopology_P2*)0, topo, field, func) &&

      !dispatch_topo_field((QuadTopology*)0,    topo, field, func) &&
      !dispatch_topo_field((QuadTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field((QuadTopology_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TriTopology*)0,     topo, field, func) &&
      !dispatch_topo_field((TriTopology_P1*)0,  topo, field, func) &&
      !dispatch_topo_field((TriTopology_P2*)0,  topo, field, func))

    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}


//
// Dispatch with (topo, func).
//
template<typename Functor>
void dispatch_3d(TopologyBase *topo, Functor &func)
{
  if (!dispatch_topo_only((HexTopology*)0,    topo, func) &&
      !dispatch_topo_only((HexTopology_P1*)0, topo, func) &&
      !dispatch_topo_only((HexTopology_P2*)0, topo, func) &&
      !dispatch_topo_only((TetTopology*)0,    topo, func) &&
      !dispatch_topo_only((TetTopology_P1*)0, topo, func) &&
      !dispatch_topo_only((TetTopology_P2*)0, topo, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch_2d(TopologyBase *topo, Functor &func)
{
  if (!dispatch_topo_only((QuadTopology*)0,    topo, func) &&
      !dispatch_topo_only((QuadTopology_P1*)0, topo, func) &&
      !dispatch_topo_only((QuadTopology_P2*)0, topo, func) &&
      !dispatch_topo_only((TriTopology*)0,     topo, func) &&
      !dispatch_topo_only((TriTopology_P1*)0,  topo, func) &&
      !dispatch_topo_only((TriTopology_P2*)0,  topo, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch(TopologyBase *topo, Functor &func)
{
  if (!dispatch_topo_only((HexTopology*)0,    topo, func) &&
      !dispatch_topo_only((HexTopology_P1*)0, topo, func) &&
      !dispatch_topo_only((HexTopology_P2*)0, topo, func) &&
      !dispatch_topo_only((TetTopology*)0,    topo, func) &&
      !dispatch_topo_only((TetTopology_P1*)0, topo, func) &&
      !dispatch_topo_only((TetTopology_P2*)0, topo, func) &&

      !dispatch_topo_only((QuadTopology*)0,    topo, func) &&
      !dispatch_topo_only((QuadTopology_P1*)0, topo, func) &&
      !dispatch_topo_only((QuadTopology_P2*)0, topo, func) &&
      !dispatch_topo_only((TriTopology*)0,     topo, func) &&
      !dispatch_topo_only((TriTopology_P1*)0,  topo, func) &&
      !dispatch_topo_only((TriTopology_P2*)0,  topo, func))

    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}


//
// Dispatch with (field, func)
//

template<typename Functor>
void dispatch_3d(FieldBase *field, Functor &func)
{
  if(dynamic_cast<Field<HexScalar>*>(field) != nullptr)
  {
    Field<HexScalar>* scalar_field  = dynamic_cast<Field<HexScalar>*>(field);
    func(*scalar_field);
  }
  else
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

} // namespace dray
#endif

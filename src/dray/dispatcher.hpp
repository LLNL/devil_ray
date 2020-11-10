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
//  For some ops(eg, iso surface), it doesn't make sense
// to call a order 0 field
template<typename DerivedTopologyT, typename Functor>
void dispatch_scalar_field_min_linear(FieldBase *field, DerivedTopologyT *topo, Functor &func)
{
  using MElemT = typename DerivedTopologyT::ElementType;

  constexpr int32 SingleComp = 1;

  using ScalarElement
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), General>;
  using ScalarElement_P1
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Linear>;
  using ScalarElement_P2
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Quadratic>;

  if(dynamic_cast<Field<ScalarElement>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement>());
    Field<ScalarElement>* scalar_field  = dynamic_cast<Field<ScalarElement>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<Field<ScalarElement_P1>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P1>());
    Field<ScalarElement_P1>* scalar_field  = dynamic_cast<Field<ScalarElement_P1>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<Field<ScalarElement_P2>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P2>());
    Field<ScalarElement_P2>* scalar_field  = dynamic_cast<Field<ScalarElement_P2>*>(field);
    func(*topo, *scalar_field);
  }
  else
    detail::cast_field_failed(field, __FILE__, __LINE__);
}


// Figure out a way to specialize based on TopoType
// No need to even call hex when its a quad topo
template<typename DerivedTopologyT, typename Functor>
void dispatch_scalar_field(FieldBase *field, DerivedTopologyT *topo, Functor &func)
{
  using MElemT = typename DerivedTopologyT::ElementType;

  constexpr int32 SingleComp = 1;

  using ScalarElement
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), General>;
  using ScalarElement_P0
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Constant>;
  using ScalarElement_P1
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Linear>;
  using ScalarElement_P2
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Quadratic>;

  if(dynamic_cast<Field<ScalarElement>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement>());
    Field<ScalarElement>* scalar_field  = dynamic_cast<Field<ScalarElement>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<Field<ScalarElement_P0>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P0>());
    Field<ScalarElement_P0>* scalar_field  = dynamic_cast<Field<ScalarElement_P0>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<Field<ScalarElement_P1>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P1>());
    Field<ScalarElement_P1>* scalar_field  = dynamic_cast<Field<ScalarElement_P1>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<Field<ScalarElement_P2>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P2>());
    Field<ScalarElement_P2>* scalar_field  = dynamic_cast<Field<ScalarElement_P2>*>(field);
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
bool dispatch_topo_field(TopologyGuessT *,
                         TopologyBase *topo,
                         FieldBase *field,
                         Functor &func)
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
bool dispatch_topo_field_min_linear(TopologyGuessT *,
                                    TopologyBase *topo,
                                    FieldBase *field,
                                    Functor &func)
{
  static_assert(!std::is_same<const TopologyGuessT*, const TopologyBase*>::value,
      "Cannot dispatch to TopologyBase. (Did you mix up tag and pointer?)");

  TopologyGuessT *derived_topo;

  if ((derived_topo = dynamic_cast<TopologyGuessT*>(topo)) != nullptr)
  {
    DRAY_INFO("Dispatched " + topo->type_name() + " topology to " + element_name<typename TopologyGuessT::ElementType>());
    dispatch_scalar_field_min_linear(field, derived_topo, func);
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


template <typename FElemGuessT, typename Functor>
bool dispatch_field_only(Field<FElemGuessT> *, FieldBase * field, Functor &func)
{
  static_assert(!std::is_same<const Field<FElemGuessT>*, const FieldBase*>::value,
      "Cannot dispatch to FieldBase. (Did you mix up tag and pointer?)");

  Field<FElemGuessT> *derived_field;

  if ((derived_field = dynamic_cast<Field<FElemGuessT>*>(field)) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<FElemGuessT>());
    func(*derived_field);
  }

  return (derived_field != nullptr);
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

// callers need at least an order 1 field
template<typename Functor>
bool dispatch_3d_min_linear(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if (!dispatch_topo_field_min_linear((HexTopology*)0,    topo, field, func) &&
      !dispatch_topo_field_min_linear((HexTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field_min_linear((HexTopology_P2*)0, topo, field, func) &&
      !dispatch_topo_field_min_linear((TetTopology*)0,    topo, field, func) &&
      !dispatch_topo_field_min_linear((TetTopology_P1*)0, topo, field, func) &&
      !dispatch_topo_field_min_linear((TetTopology_P2*)0, topo, field, func))
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
  if (!dispatch_field_only((Field<HexScalar>*)0,    field, func) &&
      !dispatch_field_only((Field<HexScalar_P1>*)0, field, func) &&
      !dispatch_field_only((Field<HexScalar_P2>*)0, field, func) &&
      !dispatch_field_only((Field<TetScalar>*)0,    field, func) &&
      !dispatch_field_only((Field<TetScalar_P1>*)0, field, func) &&
      !dispatch_field_only((Field<TetScalar_P2>*)0, field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch_2d(FieldBase *field, Functor &func)
{
  if (!dispatch_field_only((Field<QuadScalar>*)0,    field, func) &&
      !dispatch_field_only((Field<QuadScalar_P1>*)0, field, func) &&
      !dispatch_field_only((Field<QuadScalar_P2>*)0, field, func) &&
      !dispatch_field_only((Field<TriScalar>*)0,     field, func) &&
      !dispatch_field_only((Field<TriScalar_P1>*)0,  field, func) &&
      !dispatch_field_only((Field<TriScalar_P2>*)0,  field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch(FieldBase *field, Functor &func)
{
  if (!dispatch_field_only((Field<HexScalar>*)0,    field, func) &&
      !dispatch_field_only((Field<HexScalar_P1>*)0, field, func) &&
      !dispatch_field_only((Field<HexScalar_P2>*)0, field, func) &&
      !dispatch_field_only((Field<TetScalar>*)0,    field, func) &&
      !dispatch_field_only((Field<TetScalar_P1>*)0, field, func) &&
      !dispatch_field_only((Field<TetScalar_P2>*)0, field, func) &&

      !dispatch_field_only((Field<QuadScalar>*)0,    field, func) &&
      !dispatch_field_only((Field<QuadScalar_P1>*)0, field, func) &&
      !dispatch_field_only((Field<QuadScalar_P2>*)0, field, func) &&
      !dispatch_field_only((Field<TriScalar>*)0,     field, func) &&
      !dispatch_field_only((Field<TriScalar_P1>*)0,  field, func) &&
      !dispatch_field_only((Field<TriScalar_P2>*)0,  field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}


} // namespace dray
#endif

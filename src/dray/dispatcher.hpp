// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DISPATCHER_HPP
#define DRAY_DISPATCHER_HPP

#include<dray/data_model/unstructured_mesh.hpp>
#include<dray/data_model/unstructured_field.hpp>
#include<dray/error.hpp>
#include<dray/utils/data_logger.hpp>

#include <utility>
#include <type_traits>

namespace dray
{

namespace detail
{
  void cast_topo_failed(Mesh *topo, const char *file, unsigned long long line);
  void cast_field_failed(Field *field, const char *file, unsigned long long line);
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
template<typename DerivedMeshT, typename Functor>
void dispatch_scalar_field_min_linear(Field *field, DerivedMeshT *topo, Functor &func)
{
  using MElemT = typename DerivedMeshT::ElementType;

  constexpr int32 SingleComp = 1;

  using ScalarElement
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), General>;
  using ScalarElement_P1
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Linear>;
  using ScalarElement_P2
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Quadratic>;

  if(dynamic_cast<UnstructuredField<ScalarElement>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement>());
    UnstructuredField<ScalarElement>* scalar_field  = dynamic_cast<UnstructuredField<ScalarElement>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<UnstructuredField<ScalarElement_P1>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P1>());
    UnstructuredField<ScalarElement_P1>* scalar_field  = dynamic_cast<UnstructuredField<ScalarElement_P1>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<UnstructuredField<ScalarElement_P2>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P2>());
    UnstructuredField<ScalarElement_P2>* scalar_field  = dynamic_cast<UnstructuredField<ScalarElement_P2>*>(field);
    func(*topo, *scalar_field);
  }
  else
    detail::cast_field_failed(field, __FILE__, __LINE__);
}


// Figure out a way to specialize based on TopoType
// No need to even call hex when its a quad topo
template<typename DerivedMeshT, typename Functor>
void dispatch_scalar_field(Field *field, DerivedMeshT *topo, Functor &func)
{
  using MElemT = typename DerivedMeshT::ElementType;

  constexpr int32 SingleComp = 1;

  using ScalarElement
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), General>;
  using ScalarElement_P0
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Constant>;
  using ScalarElement_P1
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Linear>;
  using ScalarElement_P2
    = Element<MElemT::get_dim(), SingleComp, MElemT::get_etype(), Quadratic>;

  if(dynamic_cast<UnstructuredField<ScalarElement>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement>());
    UnstructuredField<ScalarElement>* scalar_field  = dynamic_cast<UnstructuredField<ScalarElement>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<UnstructuredField<ScalarElement_P0>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P0>());
    UnstructuredField<ScalarElement_P0>* scalar_field  = dynamic_cast<UnstructuredField<ScalarElement_P0>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<UnstructuredField<ScalarElement_P1>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P1>());
    UnstructuredField<ScalarElement_P1>* scalar_field  = dynamic_cast<UnstructuredField<ScalarElement_P1>*>(field);
    func(*topo, *scalar_field);
  }
  else if(dynamic_cast<UnstructuredField<ScalarElement_P2>*>(field) != nullptr)
  {
    DRAY_INFO("Dispatched " + field->type_name() + " field to " + element_name<ScalarElement_P2>());
    UnstructuredField<ScalarElement_P2>* scalar_field  = dynamic_cast<UnstructuredField<ScalarElement_P2>*>(field);
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

template <typename MeshGuessT, typename Functor>
bool dispatch_topo_field(MeshGuessT *,
                         Mesh *topo,
                         Field *field,
                         Functor &func)
{
  static_assert(!std::is_same<const MeshGuessT*, const Mesh*>::value,
      "Cannot dispatch to Mesh. (Did you mix up tag and pointer?)");

  MeshGuessT *derived_topo;

  if ((derived_topo = dynamic_cast<MeshGuessT*>(topo)) != nullptr)
  {
    DRAY_INFO("Dispatched " + topo->type_name() + " topology to " + element_name<typename MeshGuessT::ElementType>());
    dispatch_scalar_field(field, derived_topo, func);
  }

  return (derived_topo != nullptr);
}

template <typename MeshGuessT, typename Functor>
bool dispatch_topo_field_min_linear(MeshGuessT *,
                                    Mesh *topo,
                                    Field *field,
                                    Functor &func)
{
  static_assert(!std::is_same<const MeshGuessT*, const Mesh*>::value,
      "Cannot dispatch to Mesh. (Did you mix up tag and pointer?)");

  MeshGuessT *derived_topo;

  if ((derived_topo = dynamic_cast<MeshGuessT*>(topo)) != nullptr)
  {
    DRAY_INFO("Dispatched " + topo->type_name() + " topology to " + element_name<typename MeshGuessT::ElementType>());
    dispatch_scalar_field_min_linear(field, derived_topo, func);
  }

  return (derived_topo != nullptr);
}

template <typename MeshGuessT, typename Functor>
bool dispatch_topo_only(MeshGuessT *, Mesh *topo, Functor &func)
{
  static_assert(!std::is_same<const MeshGuessT*, const Mesh*>::value,
      "Cannot dispatch to Mesh. (Did you mix up tag and pointer?)");

  MeshGuessT *derived_topo;

  if ((derived_topo = dynamic_cast<MeshGuessT*>(topo)) != nullptr)
  {
    DRAY_INFO("Dispatched " + topo->type_name() + " topology to " + element_name<typename MeshGuessT::ElementType>());
    func(*derived_topo);
  }

  return (derived_topo != nullptr);
}


template <typename FElemGuessT, typename Functor>
bool dispatch_field_only(UnstructuredField<FElemGuessT> *, Field * field, Functor &func)
{
  static_assert(!std::is_same<const UnstructuredField<FElemGuessT>*, const Field*>::value,
      "Cannot dispatch to Field. (Did you mix up tag and pointer?)");

  UnstructuredField<FElemGuessT> *derived_field;

  if ((derived_field = dynamic_cast<UnstructuredField<FElemGuessT>*>(field)) != nullptr)
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
bool dispatch_3d(Mesh *topo, Field *field, Functor &func)
{
  if (!dispatch_topo_field((HexMesh*)0,    topo, field, func) &&
      !dispatch_topo_field((HexMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field((HexMesh_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TetMesh*)0,    topo, field, func) &&
      !dispatch_topo_field((TetMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field((TetMesh_P2*)0, topo, field, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
  return true;
}

// callers need at least an order 1 field
template<typename Functor>
bool dispatch_3d_min_linear(Mesh *topo, Field *field, Functor &func)
{
  if (!dispatch_topo_field_min_linear((HexMesh*)0,    topo, field, func) &&
      !dispatch_topo_field_min_linear((HexMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field_min_linear((HexMesh_P2*)0, topo, field, func) &&
      !dispatch_topo_field_min_linear((TetMesh*)0,    topo, field, func) &&
      !dispatch_topo_field_min_linear((TetMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field_min_linear((TetMesh_P2*)0, topo, field, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
  return true;
}

// Topologically 2d
template<typename Functor>
bool dispatch_2d(Mesh *topo, Field *field, Functor &func)
{
  if (!dispatch_topo_field((QuadMesh*)0,    topo, field, func) &&
      !dispatch_topo_field((QuadMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field((QuadMesh_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TriMesh*)0,     topo, field, func) &&
      !dispatch_topo_field((TriMesh_P1*)0,  topo, field, func) &&
      !dispatch_topo_field((TriMesh_P2*)0,  topo, field, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
  return true;
}

template<typename Functor>
void dispatch(Mesh *topo, Field *field, Functor &func)
{
  if (!dispatch_topo_field((HexMesh*)0,    topo, field, func) &&
      !dispatch_topo_field((HexMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field((HexMesh_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TetMesh*)0,    topo, field, func) &&
      !dispatch_topo_field((TetMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field((TetMesh_P2*)0, topo, field, func) &&

      !dispatch_topo_field((QuadMesh*)0,    topo, field, func) &&
      !dispatch_topo_field((QuadMesh_P1*)0, topo, field, func) &&
      !dispatch_topo_field((QuadMesh_P2*)0, topo, field, func) &&
      !dispatch_topo_field((TriMesh*)0,     topo, field, func) &&
      !dispatch_topo_field((TriMesh_P1*)0,  topo, field, func) &&
      !dispatch_topo_field((TriMesh_P2*)0,  topo, field, func))

    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}


//
// Dispatch with (topo, func).
//
template<typename Functor>
void dispatch_3d(Mesh *topo, Functor &func)
{
  if (!dispatch_topo_only((HexMesh*)0,    topo, func) &&
      !dispatch_topo_only((HexMesh_P1*)0, topo, func) &&
      !dispatch_topo_only((HexMesh_P2*)0, topo, func) &&
      !dispatch_topo_only((TetMesh*)0,    topo, func) &&
      !dispatch_topo_only((TetMesh_P1*)0, topo, func) &&
      !dispatch_topo_only((TetMesh_P2*)0, topo, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch_2d(Mesh *topo, Functor &func)
{
  if (!dispatch_topo_only((QuadMesh*)0,    topo, func) &&
      !dispatch_topo_only((QuadMesh_P1*)0, topo, func) &&
      !dispatch_topo_only((QuadMesh_P2*)0, topo, func) &&
      !dispatch_topo_only((TriMesh*)0,     topo, func) &&
      !dispatch_topo_only((TriMesh_P1*)0,  topo, func) &&
      !dispatch_topo_only((TriMesh_P2*)0,  topo, func))
    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch(Mesh *topo, Functor &func)
{
  if (!dispatch_topo_only((HexMesh*)0,    topo, func) &&
      !dispatch_topo_only((HexMesh_P1*)0, topo, func) &&
      !dispatch_topo_only((HexMesh_P2*)0, topo, func) &&
      !dispatch_topo_only((TetMesh*)0,    topo, func) &&
      !dispatch_topo_only((TetMesh_P1*)0, topo, func) &&
      !dispatch_topo_only((TetMesh_P2*)0, topo, func) &&

      !dispatch_topo_only((QuadMesh*)0,    topo, func) &&
      !dispatch_topo_only((QuadMesh_P1*)0, topo, func) &&
      !dispatch_topo_only((QuadMesh_P2*)0, topo, func) &&
      !dispatch_topo_only((TriMesh*)0,     topo, func) &&
      !dispatch_topo_only((TriMesh_P1*)0,  topo, func) &&
      !dispatch_topo_only((TriMesh_P2*)0,  topo, func))

    detail::cast_topo_failed(topo, __FILE__, __LINE__);
}


//
// Dispatch with (field, func)
//

template<typename Functor>
void dispatch_3d_scalar(Field *field, Functor &func)
{
  if (!dispatch_field_only((UnstructuredField<HexScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P2>*)0, field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch_3d_vector(Field *field, Functor &func)
{
  if (!dispatch_field_only((UnstructuredField<HexVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P2>*)0, field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch_2d(Field *field, Functor &func)
{
  if (!dispatch_field_only((UnstructuredField<QuadScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<QuadScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<QuadScalar_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar>*)0,     field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar_P1>*)0,  field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar_P2>*)0,  field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch_vector(Field *field, Functor &func)
{
  if (!dispatch_field_only((UnstructuredField<HexVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P2>*)0, field, func) &&

      !dispatch_field_only((UnstructuredField<QuadVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector>*)0,     field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_P1>*)0,  field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_P2>*)0,  field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

template<typename Functor>
void dispatch(Field *field, Functor &func)
{
  if (!dispatch_field_only((UnstructuredField<HexScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P2>*)0, field, func) &&

      !dispatch_field_only((UnstructuredField<QuadScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<QuadScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<QuadScalar_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar>*)0,     field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar_P1>*)0,  field, func) &&
      !dispatch_field_only((UnstructuredField<TriScalar_P2>*)0,  field, func) &&

      !dispatch_field_only((UnstructuredField<HexVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P2>*)0, field, func) &&

      !dispatch_field_only((UnstructuredField<QuadVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<QuadVector_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector>*)0,     field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_P1>*)0,  field, func) &&
      !dispatch_field_only((UnstructuredField<TriVector_P2>*)0,  field, func))
    detail::cast_field_failed(field, __FILE__, __LINE__);
}

// Used for mapping all fields onto the output
template<typename Functor>
void dispatch_3d(Field *field, Functor &func)
{
  if (!dispatch_field_only((UnstructuredField<HexScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexScalar_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetScalar_P2>*)0, field, func) &&

      !dispatch_field_only((UnstructuredField<HexVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<HexVector_P2>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector>*)0,    field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P1>*)0, field, func) &&
      !dispatch_field_only((UnstructuredField<TetVector_P2>*)0, field, func))
  {
    detail::cast_field_failed(field, __FILE__, __LINE__);
  }
}


} // namespace dray
#endif

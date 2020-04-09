// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ELEM_ATTR
#define DRAY_ELEM_ATTR

#include <dray/types.hpp>

#include <sstream>

namespace dray
{

enum Order
{
  General = -1,
  Constant = 0,
  Linear = 1,
  Quadratic = 2,
  Cubic = 3,
};

enum ElemType
{
  Tensor = 0u,
  Simplex = 1u
};

enum Geom
{
  Point,
  Line,
  Tri,
  Quad,
  Tet,
  Hex,
  NUM_GEOM
};


// TODO when we combine dim and etype, do that here.
//   Right now:  Shape<3, Tensor>
//   Future:     Shape<Hex>
template <int32 dim, ElemType etype>
struct Shape { };


// Define properties that are known just from shape.

// First, interface that works in both current and future system.

using ShapeTri  = Shape<2, Simplex>;
using ShapeTet  = Shape<3, Simplex>;
using ShapeQuad = Shape<2, Tensor>;
using ShapeHex  = Shape<3, Tensor>;
/// //Future:
/// using ShapeTri  = Shape<Tri>;
/// using ShapeTet  = Shape<Tet>;
/// using ShapeQuad = Shape<Quad>;
/// using ShapeHex  = Shape<Hex>;

/** get_etype() */
template <int32 dim> constexpr ElemType get_etype(Shape<dim, Simplex>) { return Simplex; }
template <int32 dim> constexpr ElemType get_etype(Shape<dim, Tensor>) { return Tensor; }

/// //Future:
/// constexpr ElemType get_etype(Shape<Tri>)  { return Simplex; }
/// constexpr ElemType get_etype(Shape<Tet>)  { return Simplex; }
/// constexpr ElemType get_etype(Shape<Quad>) { return Tensor; }
/// constexpr ElemType get_etype(Shape<Hex>)  { return Tensor; }


/** get_dim() */
template <ElemType etype> constexpr int32 get_dim(Shape<2, etype>) { return 2; }
template <ElemType etype> constexpr int32 get_dim(Shape<3, etype>) { return 3; }

/// //Future:
/// constexpr int32 get_dim(Shape<Tri>)  { return 2; }
/// constexpr int32 get_dim(Shape<Quad>) { return 2; }
/// constexpr int32 get_dim(Shape<Tet>)  { return 3; }
/// constexpr int32 get_dim(Shape<Hex>)  { return 3; }


/** get_geom() */
constexpr Geom get_geom(Shape<2, Simplex>) { return Tri; }
constexpr Geom get_geom(Shape<3, Simplex>) { return Tet; }
constexpr Geom get_geom(Shape<2, Tensor>)  { return Quad; }
constexpr Geom get_geom(Shape<3, Tensor>)  { return Hex; }

/// //Future:
/// template <Geom geom> constexpr Geom get_geom(Shape<geom>) { return geom; }



// Other properties from pos_XXX_element.tcc will be gradually migrated.

/** get_num_dofs() */
/// constexpr int32 get_num_dofs(Shape<






/// namespace specials
/// {
///   template <int32 dim, ElemType etype, int32 P>
///   struct get_num_dofs_struct_P { };
/// 
///   template <int32 dim, int32 P>
///   struct get_num_dofs_struct_P<dim, ElemType::Tensor, int32 P>
///   {
///     static constexpr int32 value = IntPow<P+1, dim>::val;
///   };
/// 
///   template <int32 P>
///   struct get_num_dofs_struct_P<2, ElemType::Simplex, int32 P>
///   {
///     static constexpr int32 value = (P+1)*(P+2)/2;
///   };
/// 
///   template <int32 P>
///   struct get_num_dofs_struct_P<3, ElemType::Simplex, int32 P>
///   {
///     static constexpr int32 value = (P+1)*(P+2)/2 * (P+3)/3;
///   };
/// 
/// 
///   template <int32 dim, ElemType etype>
///   struct get_num_dofs_struct { };
/// 
///   template <int32 dim>
///   struct get_num_dofs_struct<dim, ElemType::Tensor>
///   {
///     static constexpr int32 get(const int32 P) { return IntPow_varb<dim>::x(P+1); }
///   };
/// 
///   template <>
///   struct get_num_dofs_struct<2, ElemType::Simplex>
///   {
///     static constexpr int32 get(const int32 P) { return (P+1)*(P+2)/2; }
///   };
/// 
///   template <>
///   struct get_num_dofs_struct<3, ElemType::Simplex>
///   {
///     static constexpr int32 get(const int32 P) { return (P+1)*(P+2)/2 * (P+3)/3; }
///   };
/// }
/// 
/// // get_num_dofs()
/// template <int32 dim, ElemType etype, int32 P>
/// constexpr int32 get_num_dofs()
/// {
///   return specials::get_num_dofs_struct_P<dim, etype, P>::value;
/// }
/// 
/// // get_num_dofs()
/// template <int32 dim, ElemType etype>
/// constexpr int32 get_num_dofs(const int32 P)
/// {
///   return specials::get_num_dofs_struct<dim, etype>::get(P);
/// }


// Element attribute utils

static std::string element_type(ElemType type)
{
  if(type == ElemType::Tensor)
  {
    return "Tensor";
  }
  if(type == ElemType::Simplex)
  {
    return "Simplex";
  }
  return "unknown";
}

template<typename ElemClass>
static std::string element_name()
{
  std::stringstream ss;

  int32 dim = ElemClass::get_dim();

  if(dim == 3)
  {
    ss<<"3D"<<"_";
  }
  else if(dim == 2)
  {
    ss<<"2D"<<"_";
  }
  ss<<element_type(ElemClass::get_etype())<<"_";
  ss<<"C"<<ElemClass::get_ncomp()<<"_";
  ss<<"P"<<ElemClass::get_P();

  return ss.str();
}

template<typename ElemClass>
static std::string element_name(const ElemClass &)
{
  return element_name<ElemClass>();
}


}//namespace dray

#endif//DRAY_ELEM_ATTR

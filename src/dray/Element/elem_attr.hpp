// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ELEM_ATTR
#define DRAY_ELEM_ATTR

#include <dray/types.hpp>
#include <dray/math.hpp>
#include <dray/exports.hpp>

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

// Usage:
//   - function_of_order(OrderPolicy<2>{});         // Invokes fixed order policy overload.
//   - function_of_order(OrderPolicy<General>{2});  // Invokes general order policy overload.
//   See get_num_dofs() for an example.
template <int32 P> struct OrderPolicy { static constexpr int32 value = P; };
template <> struct OrderPolicy<General>
{
  int32 value;
};

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

namespace eattr
{

/** get_etype() */
template <int32 dim> DRAY_EXEC constexpr ElemType get_etype(Shape<dim, Simplex>) { return Simplex; }
template <int32 dim> DRAY_EXEC constexpr ElemType get_etype(Shape<dim, Tensor>) { return Tensor; }

/// //Future:
/// constexpr ElemType get_etype(Shape<Tri>)  { return Simplex; }
/// constexpr ElemType get_etype(Shape<Tet>)  { return Simplex; }
/// constexpr ElemType get_etype(Shape<Quad>) { return Tensor; }
/// constexpr ElemType get_etype(Shape<Hex>)  { return Tensor; }


/** is_etype_known() */
template <class ShapeTAG>
DRAY_EXEC constexpr bool is_etype_known(ShapeTAG)
{
  return get_etype(ShapeTAG{}) == Simplex || get_etype(ShapeTAG{}) == Tensor;
}


/** get_dim() */
template <ElemType etype> DRAY_EXEC constexpr int32 get_dim(Shape<2, etype>) { return 2; }
template <ElemType etype> DRAY_EXEC constexpr int32 get_dim(Shape<3, etype>) { return 3; }

/// //Future:
/// constexpr int32 get_dim(Shape<Tri>)  { return 2; }
/// constexpr int32 get_dim(Shape<Quad>) { return 2; }
/// constexpr int32 get_dim(Shape<Tet>)  { return 3; }
/// constexpr int32 get_dim(Shape<Hex>)  { return 3; }


/** get_geom() */
DRAY_EXEC constexpr Geom get_geom(Shape<2, Simplex>) { return Tri; }
DRAY_EXEC constexpr Geom get_geom(Shape<3, Simplex>) { return Tet; }
DRAY_EXEC constexpr Geom get_geom(Shape<2, Tensor>)  { return Quad; }
DRAY_EXEC constexpr Geom get_geom(Shape<3, Tensor>)  { return Hex; }

/// //Future:
/// template <Geom geom> constexpr Geom get_geom(Shape<geom>) { return geom; }



// Other properties from pos_XXX_element.tcc will be gradually migrated.

/** get_num_dofs() */
template <class ShapeTAG, int32 P>
DRAY_EXEC constexpr int32 get_num_dofs(ShapeTAG, OrderPolicy<P>)
{
  static_assert(is_etype_known(ShapeTAG{}), "Unknown shape type");
  static_assert(2 <= get_dim(ShapeTAG{}) && get_dim(ShapeTAG{}) <= 3, "Only 2D and 3D supported");
  return (get_etype(ShapeTAG{}) == Tensor ? IntPow<P+1, get_dim(ShapeTAG{})>::val
      :   get_geom(ShapeTAG{}) == Tri ? (P+1)*(P+2)/2
      :   get_geom(ShapeTAG{}) == Tet ? (P+1)*(P+2)/2 * (P+3)/3
      :   -1);
}

template <class ShapeTAG>
DRAY_EXEC int32 get_num_dofs(ShapeTAG, OrderPolicy<General> order_p)
{
  static_assert(is_etype_known(ShapeTAG{}), "Unknown shape type");
  static_assert(2 <= get_dim(ShapeTAG{}) && get_dim(ShapeTAG{}) <= 3, "Only 2D and 3D supported");
  const int32 p = order_p.value;
  return (get_etype(ShapeTAG{}) == Tensor ? IntPow_varb<get_dim(ShapeTAG{})>::x(p+1)
      :   get_geom(ShapeTAG{}) == Tri ? (p+1)*(p+2)/2
      :   get_geom(ShapeTAG{}) == Tet ? (p+1)*(p+2)/2 * (p+3)/3
      :   -1);
}

template <int32 P>
DRAY_EXEC constexpr int32 get_order(OrderPolicy<P>) { return P; }

template <>
DRAY_EXEC int32 get_order(OrderPolicy<General> order_p) { return order_p.value; }

template <int32 P>
DRAY_EXEC constexpr int32 get_policy_id(OrderPolicy<P>) { return P; }

template <int32 P>
DRAY_EXEC OrderPolicy<P> adapt_create_order_policy(OrderPolicy<P>, int32 order)
{
  return OrderPolicy<P>{};
}

template <>
DRAY_EXEC OrderPolicy<General> adapt_create_order_policy(OrderPolicy<General>, int32 order)
{
  return OrderPolicy<General>{order};
}

} // eattr




namespace hex_props
{
  // Edges shall be numbered first by parallel axis, then by offset.
  // Offsets are in the 2 least significant bits.
  //   X-parallel  00--03 -->  00 + (Y==1?1:0) + (Z==1?2:0)
  //   Y-parallel  04--07 -->  04 + (X==1?1:0) + (Z==1?2:0)
  //   Z-parallel  08--11 -->  08 + (X==1?1:0) + (Y==1?2:0)
  enum EdgeIds { eParX00=0,  eParX01=1,  eParX02=2,  eParX03=3,
                 eParY00=4,  eParY01=5,  eParY02=6,  eParY03=7,
                 eParZ00=8,  eParZ01=9,  eParZ02=10, eParZ03=11 };

  constexpr uint8 EdgeAxisMask   = (1u<<3) | (1u<<2);
  constexpr uint8 EdgeOffsetMask = (1u<<1) | (1u<<0);

  constexpr uint8 hex_eaxis(const uint8 eid)
  {
    return eid >> 2;
  }
  constexpr int32 hex_estride(const uint8 eid, const int32 len)
  {
    return ((eid & EdgeAxisMask) == eParX00 ? 1 : (eid & EdgeAxisMask) == eParY00 ? len : len * len);
  }
  constexpr int32 hex_eoffset0(const uint8 eid)
  {
    return ((eid & EdgeAxisMask) == eParY00 || (eid & EdgeAxisMask) == eParZ00) && (eid & (1u<<0));
  }
  constexpr int32 hex_eoffset1(const uint8 eid)
  {
    return ((eid & EdgeAxisMask) == eParX00 && (eid & (1u<<0))) || ((eid & EdgeAxisMask) == eParZ00 && (eid & (1u<<1)));
  }
  constexpr int32 hex_eoffset2(const uint8 eid)
  {
    return ((eid & EdgeAxisMask) == eParX00 || (eid & EdgeAxisMask) == eParY00) && (eid & (1u<<1));
  }


  // Faces shall be numbered first by perpendicular axis, then by offset.
  // Offsets are in the least significant bit.
  //   X-perp 00--01  -->  loX==00   hiX==01
  //   Y-perp 02--03  -->  loY==02   hiY==03
  //   Z-perp 04--05  -->  loZ==04   hiZ==05
  enum FaceIds { fPerpX00 = 0, fPerpX01 = 1,
                 fPerpY00 = 2, fPerpY01 = 3,
                 fPerpZ00 = 4, fPerpZ01 = 5 };

  constexpr uint8 FaceAxisMask   = (1u<<2) | (1u<<1);
  constexpr uint8 FaceOffsetMask = (1u<<0);

  constexpr int32 hex_fstrideU(const uint8 fid, const int32 len)
  {
    return ((fid & FaceAxisMask) == fPerpX00 ? len : 1);
  }
  constexpr int32 hex_fstrideV(const uint8 fid, const int32 len)
  {
    return ((fid & FaceAxisMask) == fPerpZ00 ? len : len*len);
  }
}










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

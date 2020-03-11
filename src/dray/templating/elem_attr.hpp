// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/types.hpp>

namespace dray
{


enum GeomEnum
{
  Line = 0u,     // Only makes sense if dim==1.

  Quad = 1u,     // Only makes sense if dim==2.
  Tri = 2u,      //

  Hex = 3u,      // Only makes sense if dim==3.
  Tet = 4u,      //
  Prism = 5u     //
};

//TODO macro-ified automatic element type naming.

/**
 * FixedDim,   WrapDim,
 * FixedNcomp, WrapNcomp,
 * FixedOrder, WrapOrder,
 * FixedGeom,  WrapGeom
 *
 * The fixed versions are templated on the fixed value.
 * static constexpr member `m` can be accessed as a class member or object member.
 *
 * The non-fixed versions contain a single member variable to carry the non-fixed value.
 * You have to take care of setting it yourself. Aggregate initialization should work.
 * Member `m` is a variable that can only be accessed as an object member.
 *
 * This interface enables an optimization pattern that avoids
 * overheads of `mov` and `call` when using the Fixed type:
 *     // -------------------
 *     template <class AttrType>
 *     void some_function(const AttrType &attr)
 *     {
 *         for (int i = 0; i < attr.m; i++)
 *             do_something(i);
 *     }
 *     // -------------------
 */

#define CREATE_ATTR(T, Attr) \
  template <T attr_val> \
  struct Fixed##Attr \
  { \
    using type = T; \
    static constexpr bool is_fixed = true; \
    static constexpr T m = attr_val; \
  }; \
   \
  struct Wrap##Attr \
  { \
    using type = T; \
    static constexpr bool is_fixed = false; \
    T m; \
  };

CREATE_ATTR(int32, Dim)
CREATE_ATTR(int32, Ncomp)
CREATE_ATTR(int32, Order)
CREATE_ATTR(uint32, Geom)

#undef CREATE_ATTR


/**
 * ElemAttr
 *
 * @brief Combines attributes into a single type. Access each attribute directly.
 *
 * Exposes e.g. typename TypeOfDim, bool fixed_dim, get_fixed_dim(), set_dim(), get_dim().
 * Substitute Dim/dim with another attribute.
 *
 * Only use the fixed interface if the attribute is one of the fixed versions.
 * Likewise, only use the nonfixed interface if the attribute is non-fixed.
 */

template <typename DimT, typename NcompT, typename OrderT, typename GeomT>
struct ElemAttr
{
  DimT dim;
  NcompT ncomp;
  OrderT order;
  GeomT geom;
};


using DefaultElemAttr = ElemAttr<WrapDim, WrapNcomp, WrapOrder, WrapGeom>;

/**
 * AttrTypeMatrix
 *
 * @brief Use it to create a one-off type from an existing ElemAttr template instantiation.
 *
 * To add a new attribute, need to add it to each existing attribute setter,
 * as well as create a new setter.
 *
 * The amount of code needed to maintain AttrTypeMatrix grows
 * as the square of the number of attributes.
 */
namespace attr_type_matrix
{
  template <typename OrigElemAttr, typename DimT> struct SetDimT {};
  template <typename OrigElemAttr, typename NcompT> struct SetNcompT {};
  template <typename OrigElemAttr, typename OrderT> struct SetOrderT {};
  template <typename OrigElemAttr, typename GeomT> struct SetGeomT {};

  template <typename DimT, typename NcompT, typename OrderT, typename GeomT, typename NewDimT>
  struct SetDimT<ElemAttr<DimT, NcompT, OrderT, GeomT>, NewDimT> {
    using type = ElemAttr<NewDimT, NcompT, OrderT, GeomT>;
  };

  template <typename DimT, typename NcompT, typename OrderT, typename GeomT, typename NewNcompT>
  struct SetNcompT<ElemAttr<DimT, NcompT, OrderT, GeomT>, NewNcompT> {
    using type = ElemAttr<DimT, NewNcompT, OrderT, GeomT>;
  };

  template <typename DimT, typename NcompT, typename OrderT, typename GeomT, typename NewOrderT>
  struct SetOrderT<ElemAttr<DimT, NcompT, OrderT, GeomT>, NewOrderT> {
    using type = ElemAttr<DimT, NcompT, NewOrderT, GeomT>;
  };

  template <typename DimT, typename NcompT, typename OrderT, typename GeomT, typename NewGeomT>
  struct SetGeomT<ElemAttr<DimT, NcompT, OrderT, GeomT>, NewGeomT> {
    using type = ElemAttr<DimT, NcompT, OrderT, NewGeomT>;
  };

}//namespace attr_type_matrix

template <typename OrigElemAttr, typename DimT>
using SetDimT = typename attr_type_matrix::SetDimT<OrigElemAttr, DimT>::type;

template <typename OrigElemAttr, typename NcompT>
using SetNcompT = typename attr_type_matrix::SetNcompT<OrigElemAttr, NcompT>::type;

template <typename OrigElemAttr, typename OrderT>
using SetOrderT = typename attr_type_matrix::SetOrderT<OrigElemAttr, OrderT>::type;

template <typename OrigElemAttr, typename GeomT>
using SetGeomT = typename attr_type_matrix::SetGeomT<OrigElemAttr, GeomT>::type;


/**
 * Some commonly used element attribute settings.
 */



}//namespace dray

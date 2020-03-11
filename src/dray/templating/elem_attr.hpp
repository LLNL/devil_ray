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
 * Method get_fixed() is exposed.
 *
 * The non-fixed versions carry the non-fixed value in a member variable.
 * Methods get() and set() are exposed.
 */

#define CREATE_ATTR(T, Attr) \
  template <T attr_val> \
  struct Fixed##Attr \
  { \
    using type = T; \
    static constexpr bool fixed = true; \
    static constexpr T get_fixed() { return attr_val; } \
   \
    T get() const { throw; } \
    void set(T) { throw; } \
  }; \
   \
  struct Wrap##Attr \
  { \
    using type = T; \
    static constexpr bool fixed = false; \
    static constexpr T get_fixed() { return -1; } \
   \
    T m_attr_val; \
    T get() const { return m_attr_val; } \
    void set(T attr_val) { m_attr_val = attr_val; } \
  };

CREATE_ATTR(int32, Dim)
CREATE_ATTR(int32, Ncomp)
CREATE_ATTR(int32, Order)
CREATE_ATTR(uint32, Geom)

#undef CREATE_ATTR


/**
 * ElemAttr
 *
 * @brief Combines attributes into a single type.
 *
 * Exposes e.g. typename TypeOfDim, bool fixed_dim, get_fixed_dim(), set_dim(), get_dim().
 * Substitute Dim/dim with another attribute.
 *
 * Only use the fixed interface if the attribute is one of the fixed versions.
 * Likewise, only use the nonfixed interface if the attribute is non-fixed.
 */

#define COMPOSE_ATTR(Attr, attr) \
  using TypeOf##Attr = Attr##T; \
  static constexpr bool fixed_##attr = Attr##T::fixed; \
  static constexpr typename Attr##T::type get_fixed_##attr() { return Attr##T::get_fixed(); } \
  \
  Attr##T m_##attr; \
  void set_##attr(typename Attr##T::type new_val) { m_##attr.set(new_val); } \
  typename Attr##T::type get_##attr() const { return m_##attr.get(); } \

template <typename DimT, typename NcompT, typename OrderT, typename GeomT>
struct ElemAttr
{
  COMPOSE_ATTR(Dim, dim)
  COMPOSE_ATTR(Ncomp, ncomp)
  COMPOSE_ATTR(Order, order)
  COMPOSE_ATTR(Geom, geom)
};

#undef COMPOSE_ATTR


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

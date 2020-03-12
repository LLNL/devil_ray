// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/types.hpp>

#include <iostream>

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
    \
    Fixed##Attr() {} \
    Fixed##Attr(T) {} \
    \
    template <typename OtherAttrT> \
    explicit Fixed##Attr(const OtherAttrT &) {} \
  }; \
   \
  struct Wrap##Attr \
  { \
    using type = T; \
    static constexpr bool is_fixed = false; \
    T m; \
    \
    Wrap##Attr() : m(T(-1)) {} \
    Wrap##Attr(T newval) : m(newval) {} \
    \
    template <typename OtherAttrT> \
    explicit Wrap##Attr(const OtherAttrT & other) : m(other.m) {} \
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
 * Dispatch: Choose a templated execution path depending on attribute data value.
 */

// Assuming a functor like this:
//
//     struct MyOpFunctor
//     {
//         template <typename NewT1, typename NewT2>  // Multiple template parameters ok, no state
//         static void x(void *data, const NewT1 &a1, const NewT2 &a2)
//         {
//             // cast *data appropriately.
//             // use the new constexpr wrapper types NewT1 and NewT2.
//         }
//     };

namespace dispatch
{

  template <class FunctorT>
  struct FixOrder2
  {
    template <typename T1, typename T2, typename ...Optional>
    static void x(void *data, const T1 &a1, const T2 &a2, Optional... optional)
    {
      if (T2::is_fixed)
        FunctorT::x(data, a1, a2, optional...);
      else
        switch(a2.m)
        {
          case 0: FunctorT::x(data, a1, FixedOrder<0>(a2), optional...); break;
          case 1: FunctorT::x(data, a1, FixedOrder<1>(a2), optional...); break;
          case 2: FunctorT::x(data, a1, FixedOrder<2>(a2), optional...); break;
          case 3: FunctorT::x(data, a1, FixedOrder<3>(a2), optional...); break;
          case 4: FunctorT::x(data, a1, FixedOrder<4>(a2), optional...); break;
          default: std::cerr << "Warning: FixOrder2 could not fix value " << a2.m << " to constexpr.\n";
              FunctorT::x(data, a1, a2, optional...); break;
        }
    }
  };

  template <class FunctorT>
  struct FixDim1
  {
    template <typename T1, typename ...Optional>
    static void x(void *data, const T1 &a1, Optional... optional)
    {
      if (T1::is_fixed)
        FunctorT::x(data, a1, optional...);
      else
        switch(a1.m)
        {
          case 1: FunctorT::x(data, FixedDim<1>(a1), optional...); break;
          case 2: FunctorT::x(data, FixedDim<2>(a1), optional...); break;
          case 3: FunctorT::x(data, FixedDim<2>(a1), optional...); break;
          default: std::cerr << "Warning: FixDim1 could not fix value " << a1.m << " to constexpr.\n";
              FunctorT::x(data, a1, optional...); break;
        }
    }
  };

}

/**
 * Some commonly used element attribute settings.
 */



}//namespace dray

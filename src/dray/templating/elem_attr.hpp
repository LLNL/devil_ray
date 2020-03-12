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


#define CREATE_ATTR(T, Attr, attr) \
  template <T attr_val> \
  struct Fixed##Attr \
  { \
    using type = T; \
    static constexpr bool is_fixed = true; \
    static constexpr T m = attr_val; \
    \
    Fixed##Attr() : attr{*this} {} \
    Fixed##Attr(T) : attr{*this} {}  \
    \
    template <typename OtherAttrT> \
    explicit Fixed##Attr(const OtherAttrT &) : attr{*this} {} \
    \
    Fixed##Attr & attr; \
  }; \
  \
  \
  struct Wrap##Attr \
  { \
    using type = T; \
    static constexpr bool is_fixed = false; \
    T m; \
    \
    Wrap##Attr() : m(T(-1)), attr{*this} {} \
    Wrap##Attr(T newval) : m(newval), attr{*this} {} \
    \
    template <typename OtherAttrT> \
    explicit Wrap##Attr(const OtherAttrT & other) : m(other.m), attr{*this} {} \
    \
    Wrap##Attr & attr; \
  }; \
  \
  template <T attr_val> \
  std::ostream & operator<<(std::ostream &out, const Fixed##Attr<attr_val> & a) { out << #Attr << a.m; return out; } \
  std::ostream & operator<<(std::ostream &out, const Wrap##Attr & a) { out << #Attr << a.m; return out; } \
  \
  namespace dbg { \
    template <T attr_val> \
    void print_fixed(std::ostream &out, const Fixed##Attr<attr_val> & a) { out << #Attr << "+"; } \
    void print_fixed(std::ostream &out, const Wrap##Attr & a) { out << #Attr << "-"; } \
  }

CREATE_ATTR(int32, Dim, dim)
CREATE_ATTR(int32, Ncomp, ncomp)
CREATE_ATTR(int32, Order, order)
CREATE_ATTR(uint32, Geom, geom)

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
  // All sub attritube member names should match what's given to CREATE_ATTR.

  ElemAttr() = default;

  template <typename OtherElemAttrT>
  explicit ElemAttr(const OtherElemAttrT &a) :
      dim{a.dim},
      ncomp{a.ncomp},
      order{a.order},
      geom{a.geom}
  {}
};

template <typename DimT, typename NcompT, typename OrderT, typename GeomT>
std::ostream &
operator<<( std::ostream &out,
            const ElemAttr<DimT, NcompT, OrderT, GeomT> & a)
{
  out << a.dim << "_" << a.ncomp << "_" << a.order << "_" << a.geom;
}

namespace dbg {
  template <typename DimT, typename NcompT, typename OrderT, typename GeomT>
  void print_fixed(std::ostream &out,
                   const ElemAttr<DimT, NcompT, OrderT, GeomT> & a)
  {
    print_fixed(out, a.dim);    out << "_";
    print_fixed(out, a.ncomp);  out << "_";
    print_fixed(out, a.order);  out << "_";
    print_fixed(out, a.geom);
  }
}


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
  // Default: Replacement (allows SetDimT<WrapDimT, FixedDimT<3>>).
  template <typename OrigElemAttr, typename DimT> struct SetDimT     { using type = DimT; };
  template <typename OrigElemAttr, typename NcompT> struct SetNcompT { using type = NcompT; };
  template <typename OrigElemAttr, typename OrderT> struct SetOrderT { using type = OrderT; };
  template <typename OrigElemAttr, typename GeomT> struct SetGeomT   { using type = GeomT; };

  // Specialize: Replace a single attribute inside the compound attribute.

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
//
// you can harden one or more attributes before calling MyOpFunctor::x(), like so:
//
//     // Non-fixed versions, initialized from some data upstream.
//     a0.dim.m = 3;
//     a1.order.m = 2;
//
//     // Calls MyOpFunctor::x() with correct constexpr template arguments.
//     FixDim0< FixOrder1< MyOpFunctor >>::x(nullptr, a0, a1);
//


namespace dispatch
{

  // "X macros" that define, for each attribute,
  // which values can be hardened by dispatch into constexpr.
#define HARD_VALS_Dim   SUB(1) SUB(2) SUB(3)
#define HARD_VALS_Ncomp SUB(1) SUB(2) SUB(3)
#define HARD_VALS_Order SUB(1) SUB(2) SUB(3)
#define HARD_VALS_Geom  SUB(Line) SUB(Quad) SUB(Tri) SUB(Hex) SUB(Tet) SUB(Prism)


  // -----------------------------------------------
  // FixDim0, FixNcomp0, FixOrder0, FixGeom0
  // -----------------------------------------------
#define SUB(cval) case cval: FunctorT::x(data, SetATTRT<T0, FixedATTR<cval>>(a0), optional...); break;
#define CREATE_FixAttr0(Attr, attr) \
  template <class FunctorT> \
  struct Fix##Attr##0 \
  { \
    template <typename T0, typename ...Optional> \
    static void x(void *data, const T0 &a0, Optional... optional) \
    { \
      if (a0.attr.is_fixed) \
        FunctorT::x(data, a0, optional...); \
      else \
        switch(a0.attr.m) \
        { \
          HARD_VALS_##Attr \
          default: std::cerr << "Warning: Fix" #Attr "0 could not fix value " << a0.attr.m << " to constexpr.\n"; \
              FunctorT::x(data, a0, optional...); break; \
        } \
    } \
  };

#define SetATTRT SetDimT
#define FixedATTR FixedDim
CREATE_FixAttr0(Dim, dim)
#undef FixedATTR
#undef SetATTRT

#define SetATTRT SetNcompT
#define FixedATTR FixedNcomp
CREATE_FixAttr0(Ncomp, ncomp)
#undef FixedATTR
#undef SetATTRT

#define SetATTRT SetOrderT
#define FixedATTR FixedOrder
CREATE_FixAttr0(Order, order)
#undef FixedATTR
#undef SetATTRT

#define SetATTRT SetGeomT
#define FixedATTR FixedGeom
CREATE_FixAttr0(Geom, geom)
#undef FixedATTR
#undef SetATTRT

#undef SUB
  // -----------------------------------------------


  // -----------------------------------------------
  // FixDim1, FixNcomp1, FixOrder1, FixGeom1
  // -----------------------------------------------
#define SUB(cval) case cval: FunctorT::x(data, a0, SetATTRT<T1, FixedATTR<cval>>(a1), optional...); break;
#define CREATE_FixAttr1(Attr, attr) \
  template <class FunctorT> \
  struct Fix##Attr##1 \
  { \
    template <typename T0, typename T1, typename ...Optional> \
    static void x(void *data, const T0 &a0, const T1 &a1, Optional... optional) \
    { \
      if (a1.attr.is_fixed) \
        FunctorT::x(data, a0, a1, optional...); \
      else \
        switch(a1.attr.m) \
        { \
          HARD_VALS_##Attr \
          default: std::cerr << "Warning: Fix" #Attr "1 could not fix value " << a1.attr.m << " to constexpr.\n"; \
              FunctorT::x(data, a0, a1, optional...); break; \
        } \
    } \
  };

#define SetATTRT SetDimT
#define FixedATTR FixedDim
CREATE_FixAttr1(Dim, dim)
#undef FixedATTR
#undef SetATTRT

#define SetATTRT SetNcompT
#define FixedATTR FixedNcomp
CREATE_FixAttr1(Ncomp, ncomp)
#undef FixedATTR
#undef SetATTRT

#define SetATTRT SetOrderT
#define FixedATTR FixedOrder
CREATE_FixAttr1(Order, order)
#undef FixedATTR
#undef SetATTRT

#define SetATTRT SetGeomT
#define FixedATTR FixedGeom
CREATE_FixAttr1(Geom, geom)
#undef FixedATTR
#undef SetATTRT

#undef SUB
  // -----------------------------------------------

  //
  // The result is a bunch of dispatcher classes similar to this:
  //
  //     template <class FunctorT>
  //     struct FixOrder1
  //     {
  //       template <typename T0, typename T1, typename ...Optional>
  //       static void x(void *data, const T0 &a0, const T1 &a1, Optional... optional)
  //       {
  //         if (a1.order.is_fixed)   // No need to harden. Use as-is.
  //           FunctorT::x(data, a0, a1, optional...);
  //         else
  //           switch(a1.m)
  //           {                 // Harden attribute 1.
  //                             // Let attribute 0 and others pass through.
  //
  //             case 0: FunctorT::x(data, a0, SetOrderT<T1, FixedOrder<0>>(a1), optional...); break;
  //             case 1: FunctorT::x(data, a0, SetOrderT<T1, FixedOrder<1>>(a1), optional...); break;
  //             case 2: FunctorT::x(data, a0, SetOrderT<T1, FixedOrder<2>>(a1), optional...); break;
  //             case 3: FunctorT::x(data, a0, SetOrderT<T1, FixedOrder<3>>(a1), optional...); break;
  //             default: std::cerr << "Warning: FixOrder1 could not fix value " << a1.m << " to constexpr.\n";
  //                 FunctorT::x(data, a0, a1, optional...); break;
  //           }
  //       }
  //     };
  //

}//namespace dispatch


/**
 * Some commonly used element attribute settings.
 */



}//namespace dray

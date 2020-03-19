// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SUBREF_HPP
#define DRAY_SUBREF_HPP

#include <dray/types.hpp>
#include <dray/aabb.hpp>
#include <dray/Element/elem_attr.hpp>

namespace dray
{
  // TODO stop using uint32 and make them int32
  template <uint32 dim, uint32 ncomp, ElemType etype, int32 P>
  class Element;

  // 2020-03-19  Masado Ishii
  //
  // Here begins the approach to separate layers of functionality/attributes
  // into separate files, but cover all element types in the same file.

  // Different SubRef for each element type.

  // Try to use the inheritance gimmick in lieu of alias template specialization.
  template <int32 dim, ElemType etype>
  struct SubRef {};

  template <int32 dim>
  struct SubRef<dim, ElemType::Quad> : public AABB<dim> {};

  template <int32 dim>
  struct SubRef<dim, ElemType::Tri> : public Vec<Vec<Float, dim>, dim+1> {};


  // If templates don't play nicely with bvh, use union:
  // template<dim> UnifiedSubRef{ union { QuadSubref qsubref, TriSubref tsubref }; };


  //
  // get_subref<ElemT>::type   (Type trait for SubRef)
  //
  template <class ElemT>
  struct get_subref
  {
    typedef void type;
  };
  template <int32 dim, int32 ncomp, ElemType etype, int32 P>
  struct get_subref<Element<dim, ncomp, etype, P>>
  {
    typedef SubRef<dim, etype> type;
  };


}//namespace dray

#endif//DRAY_SUBREF_HPP

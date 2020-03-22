// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ELEM_UTILS_HPP
#define DRAY_ELEM_UTILS_HPP

#include <dray/Element/element.hpp>

namespace dray
{

template <class ElemT, int32 dim> struct NDElem_
{
  using get_type =
  Element<dim, ElemT::get_ncomp (), ElemT::get_etype (), ElemT::get_P ()>;
};

//
// NDElem<>
//
// Get element type that is the same element type,
// except it has different topological dimension.
template <class ElemT, int32 dim>
using NDElem = typename NDElem_<ElemT, dim>::get_type;


// TODO could move FieldOn from field.hpp to here.

} // namespace dray

#endif // DRAY_ELEM_UTILS_HPP

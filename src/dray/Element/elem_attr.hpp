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
  Quad = 0u,
  Tri = 1u
};


// TODO absorb shapes.hpp, convert all tags to ITag (template_tags.hpp)
template <ElemType etype>
struct ElemTypeTag {};

/// namespace specials
/// {
///   template <int32 dim, ElemType etype, int32 P>
///   struct get_num_dofs_struct_P { };
/// 
///   template <int32 dim, int32 P>
///   struct get_num_dofs_struct_P<dim, ElemType::Quad, int32 P>
///   {
///     static constexpr int32 value = IntPow<P+1, dim>::val;
///   };
/// 
///   template <int32 P>
///   struct get_num_dofs_struct_P<2, ElemType::Tri, int32 P>
///   {
///     static constexpr int32 value = (P+1)*(P+2)/2;
///   };
/// 
///   template <int32 P>
///   struct get_num_dofs_struct_P<3, ElemType::Tri, int32 P>
///   {
///     static constexpr int32 value = (P+1)*(P+2)/2 * (P+3)/3;
///   };
/// 
/// 
///   template <int32 dim, ElemType etype>
///   struct get_num_dofs_struct { };
/// 
///   template <int32 dim>
///   struct get_num_dofs_struct<dim, ElemType::Quad>
///   {
///     static constexpr int32 get(const int32 P) { return IntPow_varb<dim>::x(P+1); }
///   };
/// 
///   template <>
///   struct get_num_dofs_struct<2, ElemType::Tri>
///   {
///     static constexpr int32 get(const int32 P) { return (P+1)*(P+2)/2; }
///   };
/// 
///   template <>
///   struct get_num_dofs_struct<3, ElemType::Tri>
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
  if(type == ElemType::Quad)
  {
    return "Quad";
  }
  if(type == ElemType::Tri)
  {
    return "Tri";
  }
  return "unknown";
}

template<typename ElemType>
static std::string element_name(ElemType)
{
  std::stringstream ss;

  int32 dim = ElemType::get_dim();

  if(dim == 3)
  {
    ss<<"3D"<<"_";
  }
  else if(dim == 2)
  {
    ss<<"2D"<<"_";
  }
  ss<<element_type(ElemType::get_etype())<<"_";
  ss<<"C"<<ElemType::get_ncomp()<<"_";
  ss<<"P"<<ElemType::get_P();

  return ss.str();
}


}//namespace dray

#endif//DRAY_ELEM_ATTR

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

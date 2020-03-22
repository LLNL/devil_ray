// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/Element/element.hpp>

namespace dray
{

//
// Explicit instantiations.

template class InvertibleElement_impl<3, ElemType::Quad, Order::General>;
template class InvertibleElement_impl<3, ElemType::Tri, Order::General>;
// If fixed-order implementations are needed as well, add instantiations for them here.

template class Element<2, 1, ElemType::Quad, Order::General>;
template class Element<2, 3, ElemType::Quad, Order::General>;
template class Element<3, 1, ElemType::Quad, Order::General>;
template class Element<3, 3, ElemType::Quad, Order::General>;
template class Element<2, 1, ElemType::Tri, Order::General>;
template class Element<2, 3, ElemType::Tri, Order::General>;
template class Element<3, 1, ElemType::Tri, Order::General>;
template class Element<3, 3, ElemType::Tri, Order::General>;

} // namespace dray

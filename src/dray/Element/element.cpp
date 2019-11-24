
#include <dray/Element/element.hpp>

namespace dray
{

//
// Explicit instantiations.

template class InvertibleElement_impl<3u, ElemType::Quad, Order::General>;
template class InvertibleElement_impl<3u, ElemType::Tri, Order::General>;
// If fixed-order implementations are needed as well, add instantiations for them here.

template class Element<2u, 1u, ElemType::Quad, Order::General>;
template class Element<2u, 3u, ElemType::Quad, Order::General>;
template class Element<3u, 1u, ElemType::Quad, Order::General>;
template class Element<3u, 3u, ElemType::Quad, Order::General>;
template class Element<2u, 1u, ElemType::Tri, Order::General>;
template class Element<2u, 3u, ElemType::Tri, Order::General>;
template class Element<3u, 1u, ElemType::Tri, Order::General>;
template class Element<3u, 3u, ElemType::Tri, Order::General>;

} // namespace dray

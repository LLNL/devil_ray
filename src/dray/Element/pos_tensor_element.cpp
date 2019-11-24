#include <dray/Element/element.hpp>

namespace dray
{


// Template instantiations.
template class QuadRefSpace<2u>;
template class QuadRefSpace<3u>;


// Template instantiations for general-order quad/hex elements.
template class Element_impl<2u, 1, ElemType::Quad, Order::General>;
template class Element_impl<2u, 3, ElemType::Quad, Order::General>;
template class Element_impl<3u, 1, ElemType::Quad, Order::General>;
template class Element_impl<3u, 3, ElemType::Quad, Order::General>;
// If fixed-order implementations are needed as well, add instantiations for them here.


} // namespace dray

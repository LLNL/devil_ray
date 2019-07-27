#include <dray/Element/element.hpp>

namespace dray
{




// Template instantiations.
template class QuadRefSpace<float, 2u>;
template class QuadRefSpace<float, 3u>;
template class QuadRefSpace<double, 2u>;
template class QuadRefSpace<double, 3u>;


// Template instantiations for general-order quad/hex elements.
template class Element_impl<float32, 2u, 1, ElemType::Quad, Order::General>;
template class Element_impl<float32, 2u, 3, ElemType::Quad, Order::General>;
template class Element_impl<float32, 3u, 1, ElemType::Quad, Order::General>;
template class Element_impl<float32, 3u, 3, ElemType::Quad, Order::General>;
template class Element_impl<float64, 2u, 1, ElemType::Quad, Order::General>;
template class Element_impl<float64, 2u, 3, ElemType::Quad, Order::General>;
template class Element_impl<float64, 3u, 1, ElemType::Quad, Order::General>;
template class Element_impl<float64, 3u, 3, ElemType::Quad, Order::General>;
// If fixed-order implementations are needed as well, add instantiations for them here.






}//namespace dray

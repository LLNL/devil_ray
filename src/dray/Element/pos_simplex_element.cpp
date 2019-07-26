#include <dray/Element/pos_simplex_element.hpp>

namespace dray
{




// Template instantiations.
template class TriRefSpace<float, 2u>;
template class TriRefSpace<float, 3u>;
template class TriRefSpace<double, 2u>;
template class TriRefSpace<double, 3u>;


// Template instantiations for general-order triangle/tetrahedral elements.
template class Element_impl<float32, 2u, 1, ElemType::Tri, Order::General>;
template class Element_impl<float32, 2u, 3, ElemType::Tri, Order::General>;
template class Element_impl<float32, 3u, 1, ElemType::Tri, Order::General>;
template class Element_impl<float32, 3u, 3, ElemType::Tri, Order::General>;
template class Element_impl<float64, 2u, 1, ElemType::Tri, Order::General>;
template class Element_impl<float64, 2u, 3, ElemType::Tri, Order::General>;
template class Element_impl<float64, 3u, 1, ElemType::Tri, Order::General>;
template class Element_impl<float64, 3u, 3, ElemType::Tri, Order::General>;
// If fixed-order implementations are needed as well, add instantiations for them here.




}//namespace dray

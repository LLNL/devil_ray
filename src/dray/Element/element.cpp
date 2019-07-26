
#include <dray/Element/element.hpp>
#include <dray/Element/pos_simplex_element.hpp>
#include <dray/Element/pos_tensor_element.hpp>
    
namespace dray
{

  //
  // Explicit instantiations.

    template class InvertibleElement_impl<float32, 3u, ElemType::Quad, Order::General>;
    template class InvertibleElement_impl<float32, 3u, ElemType::Tri, Order::General>;
    template class InvertibleElement_impl<float64, 3u, ElemType::Quad, Order::General>;
    template class InvertibleElement_impl<float64, 3u, ElemType::Tri, Order::General>;
    // If fixed-order implementations are needed as well, add instantiations for them here.

    template class Element<float32, 2u, 1u, ElemType::Quad, Order::General>;
    template class Element<float32, 2u, 3u, ElemType::Quad, Order::General>;
    template class Element<float32, 3u, 1u, ElemType::Quad, Order::General>;
    template class Element<float32, 3u, 3u, ElemType::Quad, Order::General>;
    template class Element<float32, 2u, 1u, ElemType::Tri, Order::General>;
    template class Element<float32, 2u, 3u, ElemType::Tri, Order::General>;
    template class Element<float32, 3u, 1u, ElemType::Tri, Order::General>;
    template class Element<float32, 3u, 3u, ElemType::Tri, Order::General>;

    template class Element<float64, 2u, 1u, ElemType::Quad, Order::General>;
    template class Element<float64, 2u, 3u, ElemType::Quad, Order::General>;
    template class Element<float64, 3u, 1u, ElemType::Quad, Order::General>;
    template class Element<float64, 3u, 3u, ElemType::Quad, Order::General>;
    template class Element<float64, 2u, 1u, ElemType::Tri, Order::General>;
    template class Element<float64, 2u, 3u, ElemType::Tri, Order::General>;
    template class Element<float64, 3u, 1u, ElemType::Tri, Order::General>;
    template class Element<float64, 3u, 3u, ElemType::Tri, Order::General>;


  //
  // Explicit instantiations.
  /// template class Element<float32, 1,1>;
  /// template class Element<float32, 1,2>;
  /// template class Element<float32, 1,3>;
  /// template class Element<float32, 2,1>;
  /// template class Element<float32, 2,2>;
  /// template class Element<float32, 2,3>;
  template class oldelement::Element<float32, 3,1>;
  /// template class Element<float32, 3,2>;
  template class oldelement::Element<float32, 3,3>;
  /// template class Element<float64, 1,1>;
  /// template class Element<float64, 1,2>;
  /// template class Element<float64, 1,3>;
  /// template class Element<float64, 2,1>;
  /// template class Element<float64, 2,2>;
  /// template class Element<float64, 2,3>;
  template class oldelement::Element<float64, 3,1>;
  /// template class Element<float64, 3,2>;
  template class oldelement::Element<float64, 3,3>;


}

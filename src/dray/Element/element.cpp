
#include <dray/Element/element.hpp>
    
namespace dray
{
  //
  // Explicit instantiations.
  /// template class Element<float32, 1,1>;
  /// template class Element<float32, 1,2>;
  /// template class Element<float32, 1,3>;
  /// template class Element<float32, 2,1>;
  /// template class Element<float32, 2,2>;
  /// template class Element<float32, 2,3>;
  template class Element<float32, 3,1>;
  /// template class Element<float32, 3,2>;
  template class Element<float32, 3,3>;
  /// template class Element<float64, 1,1>;
  /// template class Element<float64, 1,2>;
  /// template class Element<float64, 1,3>;
  /// template class Element<float64, 2,1>;
  /// template class Element<float64, 2,2>;
  /// template class Element<float64, 2,3>;
  template class Element<float64, 3,1>;
  /// template class Element<float64, 3,2>;
  template class Element<float64, 3,3>;


}

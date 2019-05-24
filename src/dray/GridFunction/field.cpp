#include <dray/GridFunction/field.hpp>

namespace dray
{
 
  // Explicit instantiations.
  template class FieldAccess<float32, 3,1>;     template class FieldAccess<float64, 3,1>;
  template class FieldAccess<float32, 3,2>;     template class FieldAccess<float64, 3,2>;
  template class FieldAccess<float32, 3,3>;     template class FieldAccess<float64, 3,3>;



  // Explicit instantiations.
  template class Field<float32, 3,1>;     template class Field<float64, 3,1>;
  template class Field<float32, 3,2>;     template class Field<float64, 3,2>;
  template class Field<float32, 3,3>;     template class Field<float64, 3,3>;
}

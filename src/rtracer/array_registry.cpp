#include <array_registry.hpp>

#include <algorithm>
#include <iostream>

namespace rtracer
{

std::list<ArrayInternalsBase*> ArrayRegistry::m_arrays;

void 
ArrayRegistry::add_array(ArrayInternalsBase *array)
{
  m_arrays.push_front(array);
}

void 
ArrayRegistry::remove_array(ArrayInternalsBase *array)
{
  auto it = std::find_if(m_arrays.begin(), 
                         m_arrays.end(),
                         [=] (ArrayInternalsBase *other) {return other == array; });
  if (it == m_arrays.end())
  {
    std::cerr<<"Registry: cannot remove array "<<array<<"\n";
  }
  m_arrays.remove(array);
}

} // namespace rtracer

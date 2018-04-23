#ifndef RTRACER_ARRAY_REGISTRY_HPP
#define RTRACER_ARRAY_REGISTRY_HPP

#include <array_internals_base.hpp>
#include <list>

namespace rtracer
{

class ArrayRegistry
{
public:
  static void add_array(ArrayInternalsBase *array);
  static void remove_array(ArrayInternalsBase *array);
private:
  static std::list<ArrayInternalsBase*> m_arrays;
};

} // namespace rtracer
#endif

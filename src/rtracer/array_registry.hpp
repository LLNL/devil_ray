#ifndef RTRACER_ARRAY_REGISTRY_HPP
#define RTRACER_ARRAY_REGISTRY_HPP

#include <list>
#include <stddef.h>

namespace rtracer
{

class ArrayInternalsBase;

class ArrayRegistry
{
public:
  static void   add_array(ArrayInternalsBase *array);
  static void   remove_array(ArrayInternalsBase *array);
  static void   release_device_res();
  static size_t device_usage();
  static size_t host_usage();
  static int    num_arrays();
private:
  static std::list<ArrayInternalsBase*> m_arrays;
};

} // namespace rtracer
#endif

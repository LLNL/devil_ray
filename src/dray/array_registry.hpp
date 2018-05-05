#ifndef DRAY_ARRAY_REGISTRY_HPP
#define DRAY_ARRAY_REGISTRY_HPP

#include <list>
#include <stddef.h>

namespace dray
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

} // namespace dray
#endif

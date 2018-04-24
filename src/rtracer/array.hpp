#ifndef RTRACER_ARRAY_HPP
#define RTRACER_ARRAY_HPP

#include<memory>

namespace rtracer
{

// forward declaration of internals
template<typename t> class ArrayInternals;

template<typename T>
class Array
{
public:
  Array();
  ~Array();

  size_t size();
  void resize(const size_t size);
  T* get_host_ptr();
  T* get_device_ptr();
  const T* get_host_ptr_const();
  const T* get_device_ptr_const();
private:
  std::shared_ptr<ArrayInternals<T>> m_internals;
};

} // namespace rtracer
#endif


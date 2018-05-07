#include <dray/array.hpp>
#include <dray/array_internals.hpp>

namespace dray
{

template<typename T> 
Array<T>::Array()
 : m_internals(new ArrayInternals<T>())
{
  
};

template<typename T>
Array<T>::~Array()
{

}

template<typename T>
size_t Array<T>::size()
{
  return m_internals->size();
}

template<typename T>
void 
Array<T>::resize(const size_t size)
{
  m_internals->resize(size);
}

template<typename T>
T* 
Array<T>::get_host_ptr()
{
  return m_internals->get_host_ptr();
}

template<typename T>
T* 
Array<T>::get_device_ptr()
{
  return m_internals->get_device_ptr();
}

template<typename T>
const T* 
Array<T>::get_host_ptr_const()
{
  return m_internals->get_host_ptr_const();
}

template<typename T>
const T* 
Array<T>::get_device_ptr_const()
{
  return m_internals->get_device_ptr_const();
}

// Type Explicit instatiations
template class Array<int32>;
template class Array<int64>;
template class Array<float32>;
template class Array<float64>;

} // namespace dray

// Class Explicit instatiations
#include <dray/aabb.hpp>
template class dray::Array<dray::AABB>;


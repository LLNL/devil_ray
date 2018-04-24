#include<array.hpp>
#include<array_internals.hpp>

namespace rtracer
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

// Explicit instatiations
template class Array<int>;
template class Array<float>;
template class Array<double>;
}


#ifndef DRAY_ARRAY_UTILS_HPP
#define DRAY_ARRAY_UTILS_HPP

#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>

#include <cstring>

namespace dray
{

template<typename T>
void array_memset(Array<T> &array, int val)
{
  const size_t size = array.size();
#ifdef __CUDACC__
  T * ptr = array.get_device_ptr();
  cudaMemset(ptr, val, sizeof(T) * size);
#else
  T * ptr = array.get_host_ptr();
  std::memset(ptr, val, sizeof(T) * size);
#endif

}

Array<int32> array_counting(const int32 &size, 
                            const int32 &start,
                            const int32 &step)
{
  
  Array<int32> iterator;
  iterator.resize(size);
  int32 *ptr = iterator.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    ptr[i] = start + i * step;
  });

  return iterator;
}

} // namespace dray
#endif

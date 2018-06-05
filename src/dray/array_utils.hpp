#ifndef DRAY_ARRAY_UTILS_HPP
#define DRAY_ARRAY_UTILS_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>

#include <cstring>

namespace dray
{

template<typename T>
static void array_memset_zero(Array<T> &array)
{
  const size_t size = array.size();
#ifdef DRAY_CUDA_ENABLED 
  T * ptr = array.get_device_ptr();
  cudaMemset(ptr, 0, sizeof(T) * size);
#else
  T * ptr = array.get_host_ptr();
  std::memset(ptr, 0, sizeof(T) * size);
#endif

}

template<typename T, int32 S>
static void array_memset_vec(Array<Vec<T,S>> &array, const Vec<T,S> &val)
{
  
  const int32 size = array.size();

  Vec<T,S> *array_ptr = array.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    array_ptr[i] = val;
  });
}

template<typename T>
static void array_memset(Array<T> &array, const T val)
{
  
  const int32 size = array.size();

  T *array_ptr = array.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    array_ptr[i] = val;
  });
}

static
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

#ifdef DRAY_CUDA_ENABLED 
inline __device__
Vec<float32,4> const_get_vec4f(const Vec<float32,4> *const data)
{
  const float4 temp = __ldg((const float4*) data);;
  Vec<float32,4> res;
  res[0] = temp.x;
  res[1] = temp.y;
  res[2] = temp.z;
  res[3] = temp.w;
  return res;
}
#else
inline
Vec<float32,4> const_get_vec4f(const Vec<float32,4> *const data)
{
  return data[0];
}
#endif

} // namespace dray
#endif

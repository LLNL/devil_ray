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

template<typename T>
static void array_copy(Array<T> &dest, Array<T> &src)
{
 
  assert(dest.size() == src.size());

  const int32 size = dest.size();

  T *dest_ptr = dest.get_device_ptr();
  T *src_ptr = src.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    dest_ptr[i] = src_ptr[i];
  });
}

//
// this function produces a list of ids less than or equal to the input ids
// provided. 
//
// ids: an index into the 'input' array where any value in ids must be > 0 
//      and < input.size()
// 
// input: any array type that can be used with a unary functor
//
// UnaryFunctor: a unary operation that returns a boolean value. If false
//               the index from ids is removed in the output and if true 
//               the index remains. Ex functor that returns true for any
//               input value > 0.
//
template<typename T, typename X, typename UnaryFunctor>
static Array<T> compact(Array<T> &ids, Array<X> &input, UnaryFunctor)
{
  const T *ids_ptr = ids.get_device_ptr_const(); 
  const X *input_ptr = input.get_device_ptr_const(); 
  
  // avoid lambda capture issues by declaring new functor
  UnaryFunctor apply;

  const int32 size = ids.size();
  Array<uint8> flags;
  flags.resize(size);
  
  uint8 *flags_ptr = flags.get_device_ptr();

  // apply the functor to the input to generate the compact flags
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 idx = ids_ptr[i]; 
    bool flag = apply(input_ptr[idx]);
    int32 out_val = 0;

    if(flag)
    {
      out_val = 1;
    }

    flags_ptr[i] = out_val;
  });

  std::cout<<"flag done "<<"\n";
 
  Array<int32> offsets;
  offsets.resize(size);
  int32 *offsets_ptr = offsets.get_device_ptr();

  RAJA::exclusive_scan<for_policy>(flags_ptr, flags_ptr + size, offsets_ptr,
                                        RAJA::operators::plus<int32>{});
  
  int32 out_size = offsets.get_value(size-1);
  std::cout<<"in size "<<size<<" output size "<<out_size<<"\n";
  // account for the exclusive scan by adding 1 to the 
  // size if the last flag is positive
  if(flags.get_value(size-1) > 0) out_size++;
  std::cout<<"in size "<<size<<" output size "<<out_size<<"\n";

  Array<T> output;
  output.resize(out_size);
  T *output_ptr = output.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    uint8 in_flag = flags_ptr[i];
    // if the flag is valid gather the sparse intput into
    // the compact output
    if(in_flag > 0)
    {
      const int32 out_idx = offsets_ptr[i]; 
      output_ptr[out_idx] = ids_ptr[i];
    }
  });
  
  return output;
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

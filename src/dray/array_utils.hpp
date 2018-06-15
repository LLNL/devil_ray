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
// BinaryFunctor: a binary operation that returns a boolean value. If false
//               the index from ids is removed in the output and if true 
//               the index remains. Ex functor that returns true for any
//               input value > 0.
//
template<typename T, typename X, typename Y, typename BinaryFunctor>
static Array<T> compact(Array<T> &ids, Array<X> &input_x, Array<Y> &input_y, BinaryFunctor _apply)
{
  const T *ids_ptr = ids.get_device_ptr_const(); 
  const X *input_x_ptr = input_x.get_device_ptr_const(); 
  const Y *input_y_ptr = input_y.get_device_ptr_const(); 
  
  // avoid lambda capture issues by declaring new functor
  BinaryFunctor apply = _apply;

  const int32 size = ids.size();
  Array<uint8> flags;
  flags.resize(size);
  
  uint8 *flags_ptr = flags.get_device_ptr();

  // apply the functor to the input to generate the compact flags
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 idx = ids_ptr[i]; 
    bool flag = apply(input_x_ptr[idx], input_y_ptr[idx]);
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


// This method returns an array of a subset of the values from input.
// The output has the same length as indices, where each element of the output
// is drawn from input using the corresponding index in indices.
template <typename T>
static
Array<T> gather(const Array<T> input, Array<int32> indices)
{
  const int32 size_ind = indices.size();

  Array<T> output;
  output.resize(size_ind);

  const T *input_ptr = input.get_device_ptr_const();
  const int32 *indices_ptr = indices.get_device_ptr_const();
  T *output_ptr = output.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_ind), [=] DRAY_LAMBDA (int32 ii)
  {
    output_ptr[ii] = input_ptr[indices_ptr[ii]];
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

static
Array<int32> array_random(const int32 &size,
                          const uint64 &seed,
                          const int32 &modulus)
{
  // I wanted to use both 'seed' and 'sequence number' (see CUDA curand).
  // The caller provides seed, which is shared by all array elements;
  // but different array elements have different sequence numbers.
  // The sequence numbers should advance by (size) on successive calls.

  // For the serial case, I will instead use a "call number" and change the seed,
  // by feeding (given seed + call number) into the random number generator.
  // Unfortunately, two arrays each of size N will get different entries
  // than one array of size 2N.

  static uint64 call_number = 1;    // Not 0: Avoid calling srand() with 0 and then 1.
  //static uint64 sequence_start = 0;    //future: for parallel random

  // Allocate the array.
  Array<int32> rand_array;
  rand_array.resize(size);
  //int32 *ptr = rand_array.get_device_ptr();
  int32 *host_ptr = rand_array.get_host_ptr();

  // Initialize serial random number generator, then fill array.
  srand(seed + call_number);
  for (int32 i = 0; i < size; i++)
    host_ptr[i] = rand() % modulus;
  

  // TODO parallel random number generation
//  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
//  {
//    curandState_t state;
//    curand_init(seed, sequence_start + i, 0, &state);
//    ptr[i] = curand(&state);
//  });

  call_number++;
  //sequence_start += size;

  return rand_array;
}


// Inputs: Array of something convertible to bool.
//
// Outputs: Array of destination indices. ([out])
//          The size number of things that eval'd to true.
template<typename T>
static
Array<int32> array_compact_indices(const Array<T> src, int32 &out_size)
{
  const int32 in_size = src.size();

  Array<int32> dest_indices;
  dest_indices.resize(in_size);

  // Convert the source array to 0s and 1s.
  { // (Limit the scope of one-time-use array pointers.)
    const int32 *src_ptr = src.get_device_ptr_const();
    int32 *dest_indices_ptr = dest_indices.get_device_ptr();
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, in_size), [=] DRAY_LAMBDA (int32 ii)
    {
      dest_indices_ptr[ii] = (int32) (bool) src_ptr[ii];
    });
  }

  // Use an exclusive prefix sum to compute the destination indices.
  {
    int32 *dest_indices_ptr = dest_indices.get_device_ptr();
    RAJA::exclusive_scan_inplace<for_policy>(
        dest_indices_ptr,
        dest_indices_ptr + in_size,
        RAJA::operators::plus<int32>{});
  }

  // Retrieve the size of the output array.
  out_size = *(dest_indices.get_host_ptr_const() + in_size - 1) +
      ((*src.get_host_ptr_const()) ? 1 : 0);

  return dest_indices;
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

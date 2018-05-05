#include <dray/test.hpp>
#include <dray/array.hpp>
#include <RAJA/RAJA.hpp>
namespace dray
{

void Tester::raja_loop()
{
  using omp = RAJA::omp_parallel_for_exec;
  Array<int32> array;
  constexpr int len = 1000;
  array.resize(len);
  int32 *iptr = array.get_device_ptr();
  RAJA::forall<omp>(RAJA::RangeSegment(0, len), [=] (int i)
  {
    iptr[i] = i; 
  });
     
  const int32 *hptr = array.get_host_ptr_const();
  for(int i = 0; i < len; ++i) std::cout<<hptr[i]<<" ";
}

}

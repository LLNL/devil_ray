#include <rtracer/test.hpp>

#include <umpire/Umpire.hpp>

namespace rtracer
{

void Tester::raja_loop()
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator device_allocator = rm.getAllocator("HOST");
  float* my_data_device = static_cast<float*>(device_allocator.allocate(100*sizeof(float)));
}

}

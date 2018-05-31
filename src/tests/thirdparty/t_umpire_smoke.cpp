#include "gtest/gtest.h"
#include <umpire/Umpire.hpp>

TEST(umpire_smoke, umpire_allocate)
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator host_allocator = rm.getAllocator("HOST");
  void *ptr = host_allocator.allocate(4);
  host_allocator.deallocate(ptr);
}

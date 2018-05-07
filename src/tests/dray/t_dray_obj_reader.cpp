#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/io/obj_reader.hpp>

TEST(dray_test, dray_test)
{
  std::string file_name = std::string(DATA_DIR) + "conference.obj";
  std::cout<<"File name "<<file_name<<"\n";
  ObjReader reader(file_name.c_str());
  
  dray::TriangleMesh mesh;
  reader.getRawData(mesh);
 
  std::cout<<"mcoord "<<mesh.get_coords().size()<<"\n";

  dray::AABB bounds = mesh.get_bounds();
  dray::Array<dray::AABB> aabbs = mesh.get_aabbs();
}

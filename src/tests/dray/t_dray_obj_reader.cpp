#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/triangle_mesh.hpp>
#include <dray/io/obj_reader.hpp>

TEST(dray_test, dray_test)
{
  //std::string file_name = std::string(DATA_DIR) + "enzo_obj.obj";
  std::string file_name = std::string(DATA_DIR) + "conference.obj";
  std::cout<<"File name "<<file_name<<"\n";
  ObjReader reader(file_name.c_str());
  
  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  reader.getRawData(vertices, indices);

  dray::TriangleMesh mesh(vertices, indices);
 
  std::cout<<"mcoord "<<mesh.get_coords().size()<<"\n";

}

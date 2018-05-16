#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/triangle_mesh.hpp>
#include <dray/io/obj_reader.hpp>

TEST(dray_test, dray_test)
{
  //std::string file_name = std::string(DATA_DIR) + "enzo_obj.obj";
  std::string file_name = std::string(DATA_DIR) + "enzo_tiny_obj.obj";
  //std::string file_name = std::string(DATA_DIR) + "conference.obj";
  std::cout<<"File name "<<file_name<<"\n";
  ObjReader reader(file_name.c_str());
  
  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  reader.getRawData(vertices, indices);

  dray::TriangleMesh mesh(vertices, indices);
  dray::Camera camera;
  dray::Vec3f pos = dray::make_vec3f(10,10,10);
  dray::Vec3f look_at = dray::make_vec3f(5,5,5);
  camera.set_look_at(look_at);
  camera.set_pos(pos);
  //camera.set_width(1);
  //camera.set_height(1);
  camera.reset_to_bounds(mesh.get_bounds());
  dray::ray32 rays;
  std::cout<<"creating rays\n";
  camera.create_rays(rays);
  std::cout<<"intersecting rays\n";
  mesh.intersect(rays);
 
  std::cout<<"mcoord "<<mesh.get_coords().size()<<"\n";

}

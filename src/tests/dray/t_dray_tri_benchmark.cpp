#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/triangle_mesh.hpp>
#include <dray/io/obj_reader.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/utils/timer.hpp>

#define DRAY_TRIALS 1

TEST(dray_test, dray_test_unit)
{
  std::string file_name = std::string(DATA_DIR) + "unit_cube.obj";
  std::cout<<"File name "<<file_name<<"\n";

  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  dray::TriangleMesh mesh(vertices, indices);
  dray::Camera camera;
  dray::Vec3f pos = dray::make_vec3f(10,10,10);
  dray::Vec3f look_at = dray::make_vec3f(5,5,5);
  camera.set_look_at(look_at);
  camera.set_pos(pos);
  camera.reset_to_bounds(mesh.get_bounds());
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);
  std::cout<<camera.print();

  dray::Timer timer;
  for(int i = 0; i < DRAY_TRIALS; ++i)
  {
    mesh.intersect(rays);
  }

  float time = timer.elapsed();
  float ave = time / float(DRAY_TRIALS);
  float ray_size = camera.get_width() * camera.get_height();
  float rate = (ray_size / ave) / 1e6f;
  std::cout<<"Trace rate : "<<rate<<" (Mray/sec)\n";

  dray::save_depth(rays, camera.get_width(), camera.get_height());

}

TEST(dray_test, dray_test_conference)
{
  std::string file_name = std::string(DATA_DIR) + "conference.obj";
  std::cout<<"File name "<<file_name<<"\n";

  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  dray::TriangleMesh mesh(vertices, indices);
  dray::Camera camera;
  camera.set_width(1024);
  camera.set_height(1024);

  dray::Vec3f pos = dray::make_vec3f(30,19,5);
  dray::Vec3f look_at = dray::make_vec3f(0,0,0);
  dray::Vec3f up = dray::make_vec3f(0,0,1);

  camera.set_look_at(look_at);
  camera.set_pos(pos);
  camera.set_up(up);
  //camera.reset_to_bounds(mesh.get_bounds());
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);
  std::cout<<camera.print();

  dray::Timer timer;
  for(int i = 0; i < DRAY_TRIALS; ++i)
  {
    mesh.intersect(rays);
  }

  float time = timer.elapsed();
  float ave = time / float(DRAY_TRIALS);
  float ray_size = camera.get_width() * camera.get_height();
  float rate = (ray_size / ave) / 1e6f;
  std::cout<<"Trace rate : "<<rate<<" (Mray/sec)\n";

  dray::save_depth(rays, camera.get_width(), camera.get_height());

}

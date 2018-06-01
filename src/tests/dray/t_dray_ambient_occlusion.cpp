/// #include "gtest/gtest.h"
/// #include <dray/triangle_mesh.hpp>

/// TEST(dray_ambient_occlusion, dray_ambient_occlusion_basic)
/// {
///   dray::Array<dray::Vec<dray::float32,3>> dummy_array_vec;
///   dray::Array<dray::float32>              dummy_coords;
///   dray::Array<dray::int32>                dummy_indices;
///   dray::Vec<dray::float32,3>              dummy_vec;
///   dray::Ray<dray::float32>                dummy_rays;
///   dray::IntersectionContext<dray::float32>  dummy_intersection_ctx;
/// 
///   //TODO properly ininitialize the arrays/vectors so they have enough room.
///   
///   //dray::AmbientOcclusion<dray::float32>::calc_occlusion(dummy_array_vec, dummy_array_vec, 10, dummy_grayscale);
///   //dray::AmbientOcclusion<dray::float32>::calc_occlusion(dummy_rays, 10, dummy_grayscale);
///   ///dray::AmbientOcclusion<dray::float32>::gen_occlusion(dummy_vec, dummy_vec, 10, dummy_rays);
/// 
///   dray::TriangleMesh dummy_mesh(dummy_coords, dummy_indices);
///   dummy_intersection_ctx = dummy_mesh.get_intersection_context(dummy_rays);
/// 
///   dummy_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(dummy_intersection_ctx, 10, 0.1f, 10.f);
/// }





#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/triangle_mesh.hpp>
#include <dray/ambient_occlusion.hpp>
#include <dray/io/obj_reader.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/utils/timer.hpp>

#define DRAY_TRIALS 20

TEST(dray_test, dray_test_unit)
{
  // Input the data from disk.
  std::string file_name = std::string(DATA_DIR) + "unit_cube.obj";
  std::cout<<"File name "<<file_name<<"\n";
  
  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  // Build the scene/camera.
  dray::TriangleMesh mesh(vertices, indices);
  dray::Camera camera;
  dray::Vec3f pos = dray::make_vec3f(10,10,10);
  dray::Vec3f look_at = dray::make_vec3f(5,5,5);
  camera.set_look_at(look_at);
  camera.set_pos(pos);
  camera.reset_to_bounds(mesh.get_bounds());
  dray::ray32 primary_rays;
  camera.create_rays(primary_rays);
  std::cout<<camera.print();

  /// dray::Timer timer;
  /// for(int i = 0; i < DRAY_TRIALS; ++i)
  /// {
    mesh.intersect(primary_rays);
  /// }

  /// float time = timer.elapsed();
  /// float ave = time / float(DRAY_TRIALS);
  /// float ray_size = camera.get_width() * camera.get_height();
  /// float rate = (ray_size / ave) / 1e6f;
  /// std::cout<<"Trace rate : "<<rate<<" (Mray/sec)\n";

  dray::save_depth(primary_rays, camera.get_width(), camera.get_height());

}

TEST(dray_test, dray_test_conference)
{
  // Input the data from disk.
  std::string file_name = std::string(DATA_DIR) + "conference.obj";
  std::cout<<"File name "<<file_name<<"\n";
  
  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  // Build the scene/camera.
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
  dray::ray32 primary_rays;
  camera.create_rays(primary_rays);
  std::cout<<camera.print();

  /// dray::Timer timer;
  /// for(int i = 0; i < DRAY_TRIALS; ++i)
  /// {
    mesh.intersect(primary_rays);
  /// }

  /// float time = timer.elapsed();
  /// float ave = time / float(DRAY_TRIALS);
  /// float ray_size = camera.get_width() * camera.get_height();
  /// float rate = (ray_size / ave) / 1e6f;
  /// std::cout<<"Trace rate : "<<rate<<" (Mray/sec)\n";

  // Generate occlusion rays.
  dray::IntersectionContext<dray::float32> intersection_ctx = mesh.get_intersection_context(primary_rays);
  dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, 10, .000000001f, 3.0f);
 
  /// dray::save_depth(primary_rays, camera.get_width(), camera.get_height());
  dray::save_depth(occ_rays, camera.get_width(), camera.get_height());

}

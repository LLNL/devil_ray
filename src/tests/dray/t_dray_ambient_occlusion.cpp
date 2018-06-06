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

#include <iostream>
#include <fstream>

#define DRAY_TRIALS 20

//TEST(dray_test, dray_test_unit)
void cancel_test_cube()
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
  //dray::Vec3f pos = dray::make_vec3f(10,10,10);
  //dray::Vec3f pos = dray::make_vec3f(3,3,3);
  dray::Vec3f pos = dray::make_vec3f(.9,.9,.9);
  ///dray::Vec3f look_at = dray::make_vec3f(5,5,5);
  dray::Vec3f look_at = dray::make_vec3f(.5,.5,.5);
  camera.set_look_at(look_at);
  camera.set_pos(pos);

  //camera.reset_to_bounds(mesh.get_bounds());

  //camera.set_width(1024);
  //camera.set_height(1024);
  camera.set_width(500);
  camera.set_height(500);
  //camera.set_width(10);
  //camera.set_height(10);

  dray::ray32 primary_rays;
  camera.create_rays(primary_rays);  //Must be after setting camera width, height.
  std::cout<<camera.print();

  /// //DEBUG
  /// // Report face idxs and coordinates.
  /// dray::Array<dray::float32> mesh_coords = mesh.get_coords();
  /// dray::Array<dray::int32> mesh_indices = mesh.get_indices();
  /// const float *mesh_coords_ptr = mesh_coords.get_host_ptr_const();
  /// const int *mesh_indices_ptr = mesh_indices.get_host_ptr_const();
  /// dray::int32 num_tris = mesh_indices.size() / 3;
  /// std::cerr << "Mesh coordinates......" << std::endl;
  /// dray::Vec3f v[3];
  /// for (int tri_idx = 0; tri_idx < num_tris; tri_idx++)
  /// {
  ///   for (int v_idx = 0; v_idx < 3; v_idx++)
  ///   {
  ///      int v_index = mesh_indices_ptr[3*tri_idx + v_idx];
  ///      for (int c_idx = 0; c_idx < 3; c_idx++)
  ///      {
  ///        int c_index = 3*v_index + c_idx;
  ///        v[v_idx][c_idx] = mesh_coords_ptr[c_index];
  ///      }
  ///   }
  ///   std::cerr << "Triangle (" << tri_idx << ") : "
  ///             << v[0] << " " << v[1] << " " << v[2] << std::endl;
  /// }

  dray::AABB mesh_bounds = mesh.get_bounds();

  std::cerr << "The bounds are " << mesh_bounds << std::endl;

  dray::float32 mesh_scaling =
      max(max(mesh_bounds.m_x.length(),
              mesh_bounds.m_y.length()),
              mesh_bounds.m_z.length());

  mesh.intersect(primary_rays);

  dray::save_depth(primary_rays, camera.get_width(), camera.get_height());

  // Generate occlusion rays.
  dray::int32 occ_samples = 50;
  //dray::int32 occ_samples = 10;

  dray::IntersectionContext<dray::float32> intersection_ctx = mesh.get_intersection_context(primary_rays);

  //DEBUG
  const int f5 = 174820;
  const int f6 = 174680;
  const dray::int32 *dbg_hit_idx_ptr = primary_rays.m_hit_idx.get_host_ptr_const();
  const dray::int32 *dbg_pid_ptr = primary_rays.m_pixel_id.get_host_ptr_const();
  const dray::Vec3f *dbg_normal_ptr = intersection_ctx.m_normal.get_host_ptr_const();
  dray::Vec3f dbg_normal;
  for (int ii = 0; ii < primary_rays.size(); ii++)
  {
    // Find the pixels we are interested in.
    if (dbg_pid_ptr[ii] == f5)
    {
      dbg_normal = dbg_normal_ptr[ii];
      std::cerr << "F5:" << std::endl;
      std::cerr << "hit_idx == " << dbg_hit_idx_ptr[ii] << std::endl;
      std::cerr << "normal == " << "< " << dbg_normal[0] << ", " << dbg_normal[1] << ", " << dbg_normal[2] << " >" << std::endl;
      std::cerr << "pixel_id == " << dbg_pid_ptr[ii] << std::endl;
      std::cerr << "index == " << ii << std::endl;
      std::cerr << std::endl;
    }
    if (dbg_pid_ptr[ii] >= f6-5 && dbg_pid_ptr[ii] <= f6+5)
    {
      dbg_normal = dbg_normal_ptr[ii];
      std::cerr << "F6:" << std::endl;
      std::cerr << "hit_idx == " << dbg_hit_idx_ptr[ii] << std::endl;
      std::cerr << "normal == " << "< " << dbg_normal[0] << ", " << dbg_normal[1] << ", " << dbg_normal[2] << " >" << std::endl;
      std::cerr << "pixel_id == " << dbg_pid_ptr[ii] << std::endl;
      std::cerr << "index == " << ii << std::endl;
      std::cerr << std::endl;
    }
  }

  dray::Array<dray::int32> compact_indexing_array;
  //dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 2.0 * mesh_scaling, compact_indexing_array);
  dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 0.03 * mesh_scaling, compact_indexing_array);
  const dray::int32 *compact_indexing = compact_indexing_array.get_host_ptr_const();

  //dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 2.0 * mesh_scaling);
  //dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 0.03 * mesh_scaling);

  mesh.intersect(occ_rays);

  // Write out OBJ for some lines.
  const dray::Vec3f *orig_ptr = occ_rays.m_orig.get_host_ptr_const();
  const dray::Vec3f *dir_ptr = occ_rays.m_dir.get_host_ptr_const();
  const dray::float32 *dist_ptr = occ_rays.m_dist.get_host_ptr_const();
  const dray::float32 *far_ptr = occ_rays.m_far.get_host_ptr_const();
  const dray::int32 *pid_ptr = occ_rays.m_pixel_id.get_host_ptr_const();
  std::ofstream obj_output;
  obj_output.open("occ_rays.obj");
  dray::int32 test_num_bundles = 2;
  dray::int32 test_bundle_idxs[] = {compact_indexing[f5], compact_indexing[f6]};
  for (dray::int32 test_idx = 0; test_idx < test_num_bundles; test_idx++)
  {
    dray::int32 test_offset = test_bundle_idxs[test_idx] * occ_samples;

    std::cerr << "OBJ loop: pixel_id == " << pid_ptr[test_offset] << std::endl;

    for (dray::int32 i = 0; i < occ_samples; i++)
    {
      dray::int32 occ_ray_idx = i + test_bundle_idxs[test_idx] * occ_samples;

      // Get ray origin and endpoint.
      dray::Vec3f orig = orig_ptr[occ_ray_idx];
      //dray::Vec3f tip = orig_ptr[occ_ray_idx] + dir_ptr[occ_ray_idx] * dist_ptr[occ_ray_idx];    //Using hit point.
      dray::Vec3f tip = orig_ptr[occ_ray_idx] + dir_ptr[occ_ray_idx] * far_ptr[occ_ray_idx];    //Using ray "far".
      
      // Output two vertices and then connect them.
      obj_output << "v " << orig[0] << " " << orig[1] << " " << orig[2] << std::endl;
      obj_output << "v " << tip[0] << " " << tip[1] << " " << tip[2] << std::endl;
      obj_output << "l " << 2*i+1 + test_idx*occ_samples << " " << 2*i+2 + test_idx*occ_samples << std::endl;
    }
  }
  obj_output.close();
 
  dray::save_hitrate(occ_rays, occ_samples, camera.get_width(), camera.get_height());
}

//TEST(dray_test, dray_test_conference)
void cancel_test2()
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

  dray::AABB mesh_bounds = mesh.get_bounds();
  dray::float32 mesh_scaling =
      max(max(mesh_bounds.m_x.length(),
              mesh_bounds.m_y.length()),
              mesh_bounds.m_z.length());

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

  mesh.intersect(primary_rays);

  // Generate occlusion rays.
  dray::int32 occ_samples = 50;
  //dray::int32 occ_samples = 10;

  dray::IntersectionContext<dray::float32> intersection_ctx = mesh.get_intersection_context(primary_rays);

  //dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 300.0f);
  dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 0.03 * mesh_scaling);

  mesh.intersect(occ_rays);

  /// // Write out OBJ for some lines.
  /// const dray::Vec3f *orig_ptr = occ_rays.m_orig.get_host_ptr_const();
  /// const dray::Vec3f *dir_ptr = occ_rays.m_dir.get_host_ptr_const();
  /// const dray::float32 *dist_ptr = occ_rays.m_dist.get_host_ptr_const();
  /// const dray::float32 *far_ptr = occ_rays.m_far.get_host_ptr_const();
  /// std::ofstream obj_output;
  /// obj_output.open("occ_rays.obj");
  /// dray::int32 test_primary_ray_idx = 5;
  /// for (dray::int32 i = 0; i < occ_samples; i++)
  /// {
  ///   dray::int32 occ_ray_idx = i + test_primary_ray_idx * occ_samples;

  ///   // Get ray origin and endpoint.
  ///   dray::Vec3f orig = orig_ptr[occ_ray_idx];
  ///   //dray::Vec3f tip = orig_ptr[occ_ray_idx] + dir_ptr[occ_ray_idx] * dist_ptr[occ_ray_idx];    //Using hit point.
  ///   dray::Vec3f tip = orig_ptr[occ_ray_idx] + dir_ptr[occ_ray_idx] * far_ptr[occ_ray_idx];    //Using ray "far".
  ///   
  ///   // Output two vertices and then connect them.
  ///   obj_output << "v " << orig[0] << " " << orig[1] << " " << orig[2] << std::endl;
  ///   obj_output << "v " << tip[0] << " " << tip[1] << " " << tip[2] << std::endl;
  ///   obj_output << "l " << 2*i+1 << " " << 2*i+2 << std::endl;
  /// }
  /// obj_output.close();
 
  dray::save_hitrate(occ_rays, occ_samples, camera.get_width(), camera.get_height());
}


TEST(dray_test, dray_test_city)
//void cancel_test3()
{
  // Input the data from disk.
  std::string file_name = std::string(DATA_DIR) + "city_triangulated.obj";
  std::cout<<"File name "<<file_name<<"\n";
  
  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  // Build the scene/camera.
  dray::TriangleMesh mesh(vertices, indices);
  dray::Camera camera;
  camera.set_width(1024);
  camera.set_height(1024);

  dray::AABB mesh_bounds = mesh.get_bounds();
  dray::float32 mesh_scaling =
      max(max(mesh_bounds.m_x.length(),
              mesh_bounds.m_y.length()),
              mesh_bounds.m_z.length());

  dray::Vec3f pos = dray::make_vec3f(0.0f, 0.65f, -0.75f);
  dray::Vec3f look_at = dray::make_vec3f(0.52f, 0.0f, 0.35f);
  dray::Vec3f up = dray::make_vec3f(0,1,0);

  camera.set_look_at(look_at);
  camera.set_pos(pos);
  camera.set_up(up);
  camera.reset_to_bounds(mesh.get_bounds());

  dray::ray32 primary_rays;
  camera.create_rays(primary_rays);
  std::cout<<camera.print();

  mesh.intersect(primary_rays);

  // Generate occlusion rays.
  dray::int32 occ_samples = 50;

  dray::IntersectionContext<dray::float32> intersection_ctx = mesh.get_intersection_context(primary_rays);

  dray::ray32 occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 0.03 * mesh_scaling);

  mesh.intersect(occ_rays);

  dray::save_hitrate(occ_rays, occ_samples, camera.get_width(), camera.get_height());
}

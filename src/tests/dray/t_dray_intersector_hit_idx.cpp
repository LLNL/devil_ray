
#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/triangle_mesh.hpp>
#include <dray/ambient_occlusion.hpp>
#include <dray/io/obj_reader.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/utils/png_encoder.hpp>

#include <iostream>
#include <fstream>

#define DRAY_TRIALS 20


void color_map(const int hit_idx, const int img_j, float &r, float &g, float &b);


void snapshot(dray::TriangleMesh mesh, const float cam_dist, const char *out_name)
{
  dray::Camera camera;
  dray::Vec3f pos = dray::make_vec3f(1.f,1.f,1.f) * cam_dist;
  dray::Vec3f look_at = dray::make_vec3f(0.f, 0.f, 0.f);
  camera.set_look_at(look_at);
  camera.set_pos(pos);

  int img_width = 500;
  int img_height = 500;
  camera.set_width(img_width);
  camera.set_height(img_height);

  dray::ray32 primary_rays;
  camera.create_rays(primary_rays);  //Must be after setting camera width, height.
  std::cout<<camera.print();

  mesh.intersect(primary_rays);

  const int *hit_idx_ptr = primary_rays.m_hit_idx.get_host_ptr_const();
  const int *pid_ptr = primary_rays.m_pixel_id.get_host_ptr_const();
  dray::Array<float> img_buf;
  img_buf.resize(4*primary_rays.size());
  float *img_ptr = img_buf.get_host_ptr();

  for (int ray_idx = 0; ray_idx < primary_rays.size(); ray_idx++)
  {
    int offset = 4*ray_idx;

    int hit_idx = hit_idx_ptr[ray_idx];
    int pid = pid_ptr[ray_idx];

    color_map(hit_idx, pid / img_width, img_ptr[offset+0], img_ptr[offset+1], img_ptr[offset+2]);
    img_ptr[offset + 3] = 1.f;
  }

  dray::PNGEncoder encoder;
  encoder.encode(img_ptr, img_width, img_height); 
  encoder.save(out_name);
}


TEST(dray_test, dray_test_unit)
//void cancel_test_cube()
{
  // Input the data from disk.
  std::string file_name = std::string(DATA_DIR) + "unit_cube.obj";
  std::cout<<"File name "<<file_name<<"\n";
  
  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  // Build the scene.
  dray::TriangleMesh mesh(vertices, indices);
  
  // Take pictures.
  snapshot(mesh, .9f, "color_map_cube--inside.png");
  snapshot(mesh, 2.f, "color_map_cube--outside.png");
 
}




void color_map(const int hit_idx, const int img_j, float &r, float &g, float &b)
{
  const float R[12] = {1.f, 0.f, 0.f, 1.f, 1.f, 0.f};
  const float G[12] = {0.f, 1.f, 0.f, 1.f, 0.f, 1.f};
  const float B[12] = {0.f, 0.f, 1.f, 0.f, 1.f, 1.f};

  const int stripe_width = 3;

  bool in_stripe = (img_j % (3*stripe_width)) <= stripe_width;

  int sel;
  float add = 0.f;

  switch(hit_idx)
  {
  case 0: sel = 0; in_stripe = false;    break;
  case 1: sel = 1; in_stripe = false;    break;
  case 2: sel = 2; in_stripe = false;    break;
  case 3: sel = 3; in_stripe = false;    break;
  case 9: sel = 4; in_stripe = false;    break;
  case 8: sel = 5; in_stripe = false;    break;
  case 6: sel = 0; add = .3f;   break; 
  case 4: sel = 1; add = .3f;   break; 
  case 5: sel = 2; add = .3f;   break; 
  case 11: sel = 3; add = .3f;  break; 
  case 10: sel = 4; add = .3f;  break; 
  case 7: sel = 5; add = .3f;   break; 
  }

  r = (in_stripe) ? .4f * R[sel] : R[sel];
  g = (in_stripe) ? .4f * G[sel] : G[sel];
  b = (in_stripe) ? .4f * B[sel] : B[sel];

  r = min(1.f, r + add);
  g = min(1.f, g + add);
  b = min(1.f, b + add);
}





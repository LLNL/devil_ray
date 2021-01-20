// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/hdr_image_reader.hpp>

#include <dray/rendering/renderer.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_faces, dray_hdr_reader)
{
  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "hdr_image");
  remove_test_image (output_file);

  int width, height;
  dray::Array<dray::Vec<float,3>> image = dray::read_hdr_image(image_file, width, height);
  std::cout<<"Image dims ("<<width<<","<<height<<")\n";
  dray::Framebuffer fb(width,height);
  const int32 size = width * height;
  dray::Vec<float,3> * in_ptr = image.get_host_ptr();
  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();
  for(int i = 0; i < size; ++i)
  {
    dray::Vec<float,3> in_color = in_ptr[i];
    dray::Vec<float,4> out_color = {{in_color[0], in_color[1], in_color[2], 1.f}};;
    out_ptr[i] = out_color;
  }
  fb.tone_map();
  fb.save(output_file);
}

TEST (dray_faces, dray_hdr_mapping)
{
  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "hdr_ray_lookup");
  remove_test_image (output_file);

  int width, height;
  dray::Array<dray::Vec<float,3>> image = dray::read_hdr_image(image_file, width, height);
  std::cout<<"Image dims ("<<width<<","<<height<<")\n";
  const int32 size = width * height;

  dray::Vec<float,3> * in_ptr = image.get_host_ptr();

  dray::Camera camera;
  dray::AABB<3> bounds;
  dray::Vec<float,3> bmin = {{0.f,0.f,0.f}};
  dray::Vec<float,3> bmax = {{1.f,1.f,1.f}};
  bounds.include(bmin);
  bounds.include(bmax);
  int c_width = 512;
  int c_height = 512;
  const int c_size = c_width * c_height;
  camera.set_width (c_width);
  camera.set_height(c_height);
  dray::Array<dray::Ray> rays;
  camera.reset_to_bounds(bounds);
  camera.azimuth(100);
  camera.elevate(0);
  camera.set_fov(90);
  camera.create_rays(rays);
  dray::Ray * ray_ptr = rays.get_host_ptr();

  dray::Framebuffer fb(c_width,c_height);
  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();

  for(int i = 0; i < c_size; ++i)
  {
    dray::Ray ray = ray_ptr[i];
    float x = (atan2(ray.m_dir[2],ray.m_dir[0]) + dray::pi()) / (dray::pi() * 2.f);
    float y = acos(ray.m_dir[1]) / dray::pi();
    if(x > 1 || x < 0) std::cout<<"bad x "<<x<<"\n";
    if(y > 1 || y < 0) std::cout<<"bad y "<<y<<"\n";
    int xi = float(width-1) * x;
    int yi = float(height-1) * y;
    int index = yi * width + xi;

    dray::Vec<float,3> c = in_ptr[index];
    dray::Vec<float,4> color = {{c[0], c[1], c[2], 1.f}};
    out_ptr[ray.m_pixel_id] = color;
  }

  fb.tone_map();
  fb.save(output_file);
}

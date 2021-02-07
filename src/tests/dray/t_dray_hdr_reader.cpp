// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/hdr_image_reader.hpp>

#include <dray/rendering/renderer.hpp>

#include <dray/rendering/env_map.hpp>
#include <dray/rendering/device_env_map.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>
#include <dray/random.hpp>

#include <fstream>
#include <stdlib.h>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

void write_vectors(std::vector<dray::Vec<float,3>> &dirs, std::string name);

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
  int c_width = 512;
  int c_height = 512;
  const int c_size = c_width * c_height;
  camera.set_width (c_width);
  camera.set_height(c_height);
  dray::Array<dray::Ray> rays;
  camera.set_pos({{0,0,0}});
  camera.set_up({{0,0,1}});
  camera.set_look_at({{0,1,0}});
  std::cout<<camera.print();
  //camera.azimuth(100);
  //camera.elevate(0);
  camera.set_fov(90);
  camera.create_rays(rays);
  dray::Ray * ray_ptr = rays.get_host_ptr();

  dray::Framebuffer fb(c_width,c_height);
  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();

  for(int i = 0; i < c_size; ++i)
  {
    dray::Ray ray = ray_ptr[i];

    // get the textel we can see what we are sampling
    float p = atan2(ray.m_dir[1],ray.m_dir[0]);
    if(p < 0.f) p = p + 2.f *dray::pi();
    float x =  p / (dray::pi() * 2.f);
    float y = acos(dray::clamp(ray.m_dir[2],-1.f, 1.f)) / dray::pi();

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

TEST (dray_faces, dray_env_map)
{
  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "env_sample");
  remove_test_image (output_file);

  int width, height;
  dray::Array<dray::Vec<float,3>> image = dray::read_hdr_image(image_file, width, height);

  dray::Framebuffer fb(width,height);
  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();
  dray::Vec<float,3> * in_ptr = image.get_host_ptr();
  for(int i = 0; i < width * height; ++i)
  {
    dray::Vec<float,3> c = in_ptr[i];
    dray::Vec<float,4> color = {{c[0], c[1], c[2], 1.f}};
    out_ptr[i] = color;
  }

  dray::EnvMap map;
  map.load(image_file);

  int32 samples = 100000;
  dray::DeviceEnvMap d_map(map);

  dray::Array<dray::Vec<dray::uint32,2>> rstate;
  rstate.resize(1);
  bool deterministic = true;
  dray::seed_rng(rstate, deterministic);
  dray::Vec<dray::uint32,2> rand_state = rstate.get_value(0);

  for(int i = 0; i < samples; ++i)
  {
    dray::Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);

    //dray::Vec<float32,2> xy = d_map.samplei(rand);
    //int xi = float(width-1) * xy[0];
    //int yi = float(height-1) * xy[1];
    //int index = yi * width + xi;

    float32 pdf;
    dray::Vec<float32,3> dir = d_map.sample(rand,pdf);
    dray::Vec<float32,3> color = d_map.color(dir);

    // get the textel we can see what we are sampling
    float p = atan2(dir[1],dir[0]);
    if(p < 0.f) p = p + 2.f *dray::pi();
    float x =  p / (dray::pi() * 2.f);
    float y = acos(dray::clamp(dir[2],-1.f, 1.f)) / dray::pi();
    if(x > 1 || x < 0) std::cout<<"bad x "<<x<<"\n";
    if(y > 1 || y < 0) std::cout<<"bad y "<<y<<"\n";
    int xi = float(width-1) * x;
    int yi = float(height-1) * y;
    int index = yi * width + xi;

    out_ptr[index] = {{1.f, 0, 0, 1.f}};
  }

  //for(int i = 0; i < width * height; ++i)
  //{
  //  float32 value = d_map.m_distribution.m_func[i];
  //  dray::Vec<float,4> color = {{value, value, value, 1.f}};
  //  out_ptr[i] = color;
  //}

  fb.tone_map();
  fb.save(output_file);
}

TEST (dray_faces, dray_rand_test)
{
  // sanity check for random numbers
  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "rand_test");
  remove_test_image (output_file);

  int width = 1024, height = 1024;
  dray::Framebuffer fb(width,height);
  fb.clear();
  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();

  const int32 samples = 10000;

  dray::Array<dray::Vec<dray::uint32,2>> rstate;
  rstate.resize(1);
  bool deterministic = true;
  dray::seed_rng(rstate, deterministic);
  dray::Vec<dray::uint32,2> rand_state = rstate.get_value(0);

  for(int i = 0; i < samples; ++i)
  {
    dray::Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);

    // get the textel we can see what we are sampling
    int xi = float(width-1) * rand[0];
    int yi = float(height-1) * rand[1];
    int index = yi * width + xi;

    out_ptr[index] = {{1.f, 0, 0, 1.f}};
  }

  fb.composite_background();

  fb.save(output_file);
}

TEST (dray_faces, dray_distribution2d)
{
  // sanity check for random numbers
  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "dist2d_test");
  remove_test_image (output_file);

  int width = 512, height = 512;
  dray::Array<dray::float32> function;
  function.resize(width * height);

  dray::Framebuffer fb(width,height);
  fb.clear();

  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();
  float * f_ptr = function.get_host_ptr();

  for(int h = 0; h < height; ++h)
  {
    for(int w = 0; w < width; ++w)
    {
      float x = float(w) / float(width);
      float y = float(h) / float(height);
      float value= x*x + y*y;
      int index = h * width + w;
      f_ptr[index] = value;
      out_ptr[index] = {{value,value,value, 1.f}};
    }
  }

  dray::Distribution2D distribution(function,width, height);
  dray::DeviceDistribution2D d_dist(distribution);

  const int32 samples = 1000;

  dray::Array<dray::Vec<dray::uint32,2>> rstate;
  rstate.resize(1);
  bool deterministic = true;
  dray::seed_rng(rstate, deterministic);
  dray::Vec<dray::uint32,2> rand_state = rstate.get_value(0);

  std::vector<dray::Vec<float,3>> dirs;
  for(int i = 0; i < samples; ++i)
  {
    dray::Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);


    float pdf;
    dray::Vec<float32,2> sample = d_dist.sample(rand,pdf);

    // get the textel we can see what we are sampling
    int xi = float(width-1) * sample[0];
    int yi = float(height-1) * sample[1];
    int index = yi * width + xi;
    out_ptr[index] = {{1.f, 0, 0, 1.f}};

    float theta = sample[1] * dray::pi();
    float phi = sample[0] * dray::pi() * 2.f;
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    dray::Vec<float,3> dir = {{sin_theta * cos_phi,
                               sin_theta * sin_phi,
                               cos_theta}};
    dir.normalize();
    dirs.push_back(dir);
  }

  write_vectors(dirs, "env_rays");
  fb.tone_map();
  //fb.composite_background();
  fb.save(output_file);
}

void write_vectors(std::vector<dray::Vec<float32,3>> &dirs, std::string name)
{
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<int32> conn;


  x.push_back(0.f);
  y.push_back(0.f);
  z.push_back(0.f);

  int conn_count = 1;

  for(auto d : dirs)
  {
    x.push_back(d[0]);
    y.push_back(d[1]);
    z.push_back(d[2]);
    conn.push_back(0);
    conn.push_back(conn_count);
    conn_count++;
  }

  conduit::Node domain;

  domain["coordsets/coords/type"] = "explicit";
  domain["coordsets/coords/values/x"].set(x);
  domain["coordsets/coords/values/y"].set(y);
  domain["coordsets/coords/values/z"].set(z);
  domain["topologies/mesh/type"] = "unstructured";
  domain["topologies/mesh/coordset"] = "coords";
  domain["topologies/mesh/elements/shape"] = "line";
  domain["topologies/mesh/elements/connectivity"].set(conn);

  conduit::Node dataset;
  dataset.append() = domain;
  conduit::Node info;
  if(!conduit::blueprint::mesh::verify(dataset,info))
  {
    info.print();
  }
  conduit::relay::io_blueprint::save(domain, name+".blueprint_root");
}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/blueprint_low_order.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/test_renderer.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>


#include <fstream>
#include <stdlib.h>

conduit::Node cbox;

std::shared_ptr<dray::Surface>
create_quad( std::vector<double> x,
             std::vector<double> y,
             std::vector<double> z,
             float color[3])
{

  conduit::Node dataset;
  dataset["coordsets/coords/type"] = "explicit";
  dataset["coordsets/coords/values/x"].set(x);
  dataset["coordsets/coords/values/y"].set(y);
  dataset["coordsets/coords/values/z"].set(z);
  std::vector<int> conn = {0,1,2,3};

  dataset["topologies/topo/type"] = "unstructured";
  dataset["topologies/topo/coordset"] = "coords";
  dataset["topologies/topo/elements/shape"] = "quad";
  dataset["topologies/topo/elements/connectivity"].set(conn);
  std::vector<double> field = {0.};
  dataset["fields/default/association"] = "element";
  dataset["fields/default/topology"] = "topo";
  dataset["fields/default/values"].set(field);

  cbox.append() = dataset;

  dray::DataSet box_dset = dray::BlueprintLowOrder::import(dataset);
  dray::Collection col;
  col.add_domain(box_dset);

  dray::ColorTable quad_color_table;
  quad_color_table.clear();
  dray::Vec<float,3> vcolor = {{color[0],color[1],color[2]}};
  quad_color_table.add_point(0,vcolor);
  quad_color_table.add_point(1,vcolor);

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(col);
  surface->field("default");
  surface->color_map().color_table(quad_color_table);
  return surface;
}

std::shared_ptr<dray::Surface>
create_box( std::vector<double> x,
            std::vector<double> y,
            std::vector<double> z,
            float color[3])
{

  conduit::Node dataset;

  dataset["coordsets/coords/type"] = "explicit";
  dataset["coordsets/coords/values/x"].set(x);
  dataset["coordsets/coords/values/y"].set(y);
  dataset["coordsets/coords/values/z"].set(z);
  std::vector<int> conn = {0,1,2,3,
                           4,5,6,7,
                           8,9,10,11,
                           12,13,14,15,
                           16,17,18,19,
                           20,21,22,23};

  dataset["topologies/topo/type"] = "unstructured";
  dataset["topologies/topo/coordset"] = "coords";
  dataset["topologies/topo/elements/shape"] = "quad";
  dataset["topologies/topo/elements/connectivity"].set(conn);
  std::vector<double> field = {0.,0.,0.,0.,0.,0.};
  dataset["fields/default/association"] = "element";
  dataset["fields/default/topology"] = "topo";
  dataset["fields/default/values"].set(field);;
  cbox.append() = dataset;


  dray::DataSet box_dset = dray::BlueprintLowOrder::import(dataset);
  dray::Collection col;
  col.add_domain(box_dset);

  dray::ColorTable quad_color_table;
  quad_color_table.clear();
  dray::Vec<float,3> vcolor = {{color[0],color[1],color[2]}};
  quad_color_table.add_point(0,vcolor);
  quad_color_table.add_point(1,vcolor);

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(col);
  surface->field("default");
  surface->color_map().color_table(quad_color_table);
  return surface;
}

dray::Collection create_box(dray::AABB<3> bounds, float scale = 6.f)
{
  conduit::Node dataset;

  float32 dx = bounds.m_ranges[0].length() * 0.5 * scale;
  float32 dy = bounds.m_ranges[1].length() * 0.5 * scale;
  float32 dz = bounds.m_ranges[2].length() * 0.5 * scale;

  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  x.resize(8);
  y.resize(8);
  z.resize(8);

  float x_min = bounds.m_ranges[0].min() - dx;
  float x_max = bounds.m_ranges[0].max() + dx;
  float y_min = bounds.m_ranges[1].min() - dy;
  float y_max = bounds.m_ranges[1].max() + dy;
  float z_min = bounds.m_ranges[2].min() - dz;
  float z_max = bounds.m_ranges[2].max() + dz;

  x[0] = x_min;
  y[0] = y_min;
  z[0] = z_min;

  x[1] = x_max;
  y[1] = y_min;
  z[1] = z_min;

  x[2] = x_max;
  y[2] = y_max;
  z[2] = z_min;

  x[3] = x_min;
  y[3] = y_max;
  z[3] = z_min;

  x[4] = x_min;
  y[4] = y_min;
  z[4] = z_max;

  x[5] = x_max;
  y[5] = y_min;
  z[5] = z_max;

  x[6] = x_max;
  y[6] = y_max;
  z[6] = z_max;

  x[7] = x_min;
  y[7] = y_max;
  z[7] = z_max;

  dataset["coordsets/coords/type"] = "explicit";
  dataset["coordsets/coords/values/x"].set(x);
  dataset["coordsets/coords/values/y"].set(y);
  dataset["coordsets/coords/values/z"].set(z);
  std::vector<int> conn = {0,1,2,3,
                           4,5,6,7,
                           0,1,5,4,
                           1,2,6,5,
                           2,3,7,6,
                           3,0,4,7};

  dataset["topologies/topo/type"] = "unstructured";
  dataset["topologies/topo/coordset"] = "coords";
  dataset["topologies/topo/elements/shape"] = "quad";
  dataset["topologies/topo/elements/connectivity"].set(conn);
  std::vector<double> field = {0.,0.,0.,0.,0.,0.};
  dataset["fields/default/association"] = "element";
  dataset["fields/default/topology"] = "topo";
  dataset["fields/default/values"].set(field);;

  conduit::relay::io_blueprint::save(dataset, "box.blueprint_root");

  dray::DataSet box_dset = dray::BlueprintLowOrder::import(dataset);
  dray::Collection col;
  col.add_domain(box_dset);
  return col;
}

std::vector<std::shared_ptr<dray::Surface>>
create_cornel_box(std::vector<dray::Material> &materials)
{
  conduit::Node dataset;
  float white[3] = {0.73f,0.73f,0.73f};
  float green[3] = {0.12f,0.456f,0.12f};
  float red[3] = {0.65f,0.05f,0.05f};

  dray::Material diffuse;
  diffuse.m_specular = 0.f;
  diffuse.m_roughness = 1.f;

  dray::Material mix;
  mix.m_specular = 0.5f;
  mix.m_roughness = 0.2f;

  dray::Material specular;
  specular.m_specular = 1.0f;
  specular.m_ior = 1.3f;
  specular.m_spec_trans = 0.8f;
  specular.m_subsurface = 1.0f;
  specular.m_roughness = .01f;
  //specular.m_metallic = 0.5f;

  std::vector<std::shared_ptr<dray::Surface>> cornell;

  {
    // floor
    std::vector<double> x = {552.8f, 0.f, 0.f,    549.6f};
    std::vector<double> y = {0.f,    0.f, 0.f,    0.f};
    std::vector<double> z = {0.f,    0.f, 559.2f, 559.2f};

    cornell.push_back(create_quad(x,y,z, white));
    materials.push_back(diffuse);
  }

  {
    // ceiling
    std::vector<double> x = {556.0f, 556.0f, 0.f,    0.0f};
    std::vector<double> y = {548.8f, 548.8f, 548.8f, 548.8f};
    std::vector<double> z = {0.f,    559.2f, 559.2f, 0.f};
    cornell.push_back(create_quad(x,y,z, white));
    materials.push_back(diffuse);
  }

  {
    // back wall
    std::vector<double> x = {549.6f, 0.f,    0.f,    556.0f};
    std::vector<double> y = {0.f,    0.f,    548.8f, 548.8f};
    std::vector<double> z = {559.2f, 559.2f, 559.2f, 559.2f};
    cornell.push_back(create_quad(x,y,z, white));
    materials.push_back(diffuse);
  }

  {
    // right wall
    std::vector<double> x = {0.f,    0.f, 0.f,    0.f};
    std::vector<double> y = {0.f,    0.f, 548.8f, 548.8f};
    std::vector<double> z = {559.2f, 0.f, 0.f,    559.2f};
    cornell.push_back(create_quad(x,y,z, green));
    materials.push_back(diffuse);
  }
  {
    // left wall
    std::vector<double> x = {552.8f, 549.6,  556.0f, 556.0f};
    std::vector<double> y = {0.f,    0.f,    548.8f, 548.8f};
    std::vector<double> z = {0.f,    559.2f, 559.2f, 0.f};
    cornell.push_back(create_quad(x,y,z, red));
    materials.push_back(diffuse);
  }

  {
    // short box
    std::vector<double> x =
      {130.0,82.0,240.0,290.0,290.0,290.0,240.0,
       240.0,130.0,130.0,290.0,290.0,82.0,82.0,
       130.0,130.0,240.0,240.0,82.0,82.0};
    std::vector<double> y =
      {165.0,165.0,165.0,165.0,0.0,165.0,165.0,
       0.0,0.0,165.0,165.0,0.0,0.0,165.0,165.0,
       0.0,0.0,165.0,165.0,0.0};
    std::vector<double> z =
      {65.0,225.0,272.0,114.0,114.0,114.0,272.0,
       272.0,65.0,65.0,114.0,114.0,225.0,225.0,
       65.0,65.0,272.0,272.0,225.0,225.0};
    cornell.push_back(create_box(x,y,z, white));
    materials.push_back(diffuse);
  }
  {
    // tall box
    std::vector<double> x =
      {423.0,265.0,314.0,472.0,423.0,423.0,472.0,
       472.0,472.0,472.0,314.0,314.0,314.0,314.0,
       265.0,265.0,265.0,265.0,423.0,423.0};
    std::vector<double> y =
      {330.0,330.0,330.0,330.0,0.0,330.0,330.0,
       0.0,0.0,330.0,330.0,0.0,0.0,330.0,330.0,
       0.0,0.0,330.0,330.0,0.0};
    std::vector<double> z =
      {247.0,296.0,456.0,406.0,247.0,247.0,406.0,
       406.0,406.0,406.0,456.0,456.0,456.0,456.0,
       296.0,296.0,296.0,296.0,247.0,247.0};
    cornell.push_back(create_box(x,y,z, white));
    materials.push_back(specular);
  }

  conduit::relay::io::blueprint::write_mesh(cbox, "cbox", "hdf5");

  return cornell;
}


dray::TriangleLight create_cornell_light1()
{
  dray::TriangleLight light;
  //light.m_v0 = {{343.0f, 548.8f, 227.0f}};
  //light.m_v1 = {{343.0f, 548.8f, 332.0f}};
  //light.m_v2 = {{213.0f, 548.8f, 332.0f}};
  light.m_v0 = {{363.0f, 548.7f, 207.0f}};
  light.m_v1 = {{363.0f, 548.7f, 352.0f}};
  light.m_v2 = {{193.0f, 548.7f, 352.0f}};
  light.m_intensity[0] = 15.0;
  light.m_intensity[1] = 15.0;
  light.m_intensity[2] = 15.0;
  return light;
}

dray::TriangleLight create_cornell_light2()
{
  dray::TriangleLight light;
  //light.m_v0 = {{343.0f, 548.8f, 227.0f}};
  //light.m_v1 = {{213.0f, 548.8f, 332.0f}};
  //light.m_v2 = {{213.0f, 548.8f, 227.0f}};
  light.m_v0 = {{363.0f, 548.7f, 207.0f}};
  light.m_v1 = {{193.0f, 548.7f, 352.0f}};
  light.m_v2 = {{193.0f, 548.7f, 207.0f}};
  light.m_intensity[0] = 15.0;
  light.m_intensity[1] = 15.0;
  light.m_intensity[2] = 15.0;
  return light;
}

dray::SphereLight create_light(dray::Camera &camera, dray::AABB<3> bounds)
{
  dray::Vec<float,3> look_at = camera.get_look_at();
  dray::Vec<float,3> pos = camera.get_pos();
  dray::Vec<float,3> up = camera.get_up();
  up.normalize();
  dray::Vec<float,3> look = look_at - pos;
  float mag = look.magnitude();
  dray::Vec<float,3> right = cross (look, up);
  right.normalize();

  dray::Vec<float, 3> miner_up = cross (right, look);
  miner_up.normalize();

  dray::Vec<float, 3> light_pos = pos + .1f * mag * miner_up;
  dray::SphereLight light;
  light.m_pos = light_pos;
  light.m_radius = bounds.max_length() * 0.10;
  light.m_intensity[0] = 80.75;
  light.m_intensity[1] = 80.75;
  light.m_intensity[2] = 80.75;
  return light;
}

TEST (dray_test_render, dray_cornell_box)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "cornell_box");
  remove_test_image (output_file);

  // Camera
  const int c_width  = 512;
  const int c_height = 512;
  int32 samples = 5;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.set_pos({{278.f, 273.f, -1200.f}});
  camera.set_look_at({{278.f, 273.f, 800.f}});

  dray::TriangleLight light1 = create_cornell_light1();
  dray::TriangleLight light2 = create_cornell_light2();

  dray::Material mat;
  dray::TestRenderer renderer;
  std::vector<dray::Material> mats;

  auto box = create_cornel_box(mats);
  for(int i = 0; i < box.size(); ++i)
  {
    renderer.add(box[i], mats[i]);
  }


  renderer.samples(samples);
  renderer.add_light(light1);
  renderer.add_light(light2);
  dray::Framebuffer fb = renderer.render(camera);
  fb.background_color({{0.f,0.f,0.f}});
  fb.composite_background();

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file));
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}
#if 0
TEST (dray_faces, dray_impeller_faces)
{
  //std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
  std::string root_file = "/Users/larsen30/research/test_builds/devil_ray/ascent/build/utilities/replay/clipped_contour.cycle_000000.root";
  //std::string root_file = "/usr/workspace/larsen30/pascal/test_builds/dray_path/devil_ray/clipped_contour.cycle_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "impeller_faces");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);
  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(dataset);

  dray::Collection box = create_box(dataset.bounds());

  std::cout<<"Size "<<faces.local_size()<<"\n";

  dray::ColorTable color_table ("Spectral");
  //color_table.add_alpha (0.f, 0.1f);
  //color_table.add_alpha (1.0f, 0.1f);

  dray::ColorTable box_color_table;
  box_color_table.clear();
  dray::Vec<float,3> grey = {{.7f,.7f,.7f}};
  box_color_table.add_point(0,grey);
  box_color_table.add_point(1,grey);

  // Camera
  const int c_width  = 512;
  const int c_height = 512;
  int32 samples = 10;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());
  camera.azimuth(-60.f);

  dray::SphereLight light = create_light(camera, dataset.bounds());

  dray::Material mat;
  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("Ex");
  surface->color_map().color_table(color_table);

  std::shared_ptr<dray::Surface> box_s
    = std::make_shared<dray::Surface>(box);
  box_s->field("default");
  box_s->color_map().color_table(box_color_table);
  box_s->draw_mesh(true);
  box_s->mesh_sub_res(10);
  box_s->line_thickness(0.2);

  //surface->draw_mesh (true);
  //surface->line_thickness(.1);

  dray::TestRenderer renderer;
  renderer.add(surface, mat);
  renderer.add(box_s, mat);
  renderer.add_light(light);
  renderer.samples(samples);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file));
  fb.save_depth (output_file + "_depth");
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}
#endif

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
#include <dray/point_topology.hpp>
#include <dray/GridFunction/low_order_field.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/points.hpp>
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

// expects vtk ordering
std::shared_ptr<dray::Surface>
create_quad( std::vector<double> x,
             std::vector<double> y,
             std::vector<double> z,
             float color[3],
             std::string name)
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

  conduit::relay::io_blueprint::save(dataset, name + ".blueprint_root");

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

std::shared_ptr<dray::Points>
create_spheres( std::vector<double> x,
                std::vector<double> y,
                std::vector<double> z,
                std::vector<double> r,
                float color[3],
                std::string name)
{

  // blueprint cant do spheres
  dray::Array<dray::Vec<float32,3>> points;
  dray::Array<float32> radii;
  dray::Array<float32> field;
  points.resize(x.size());
  radii.resize(x.size());
  field.resize(x.size());
  dray::Vec<float32,3> *points_ptr = points.get_host_ptr();
  float32 *radii_ptr = radii.get_host_ptr();
  float32 *field_ptr = field.get_host_ptr();

  for(int i = 0; i < x.size(); ++i)
  {
    points_ptr[i][0] = x[i];
    points_ptr[i][1] = y[i];
    points_ptr[i][2] = z[i];
    radii_ptr[i] = r[i];
    field_ptr[i] = 0;
  }

  std::shared_ptr<dray::PointTopology> topo
    = std::make_shared<dray::PointTopology>(points, radii);

  dray::DataSet point_dset(topo);
  std::shared_ptr<dray::LowOrderField> lfield =
    std::make_shared<dray::LowOrderField>(field,dray::LowOrderField::Assoc::Vertex);
  lfield->name("default");
  point_dset.add_field(lfield);

  dray::Collection col;
  col.add_domain(point_dset);

  dray::ColorTable quad_color_table;
  quad_color_table.clear();
  dray::Vec<float,3> vcolor = {{color[0],color[1],color[2]}};
  quad_color_table.add_point(0,vcolor);
  quad_color_table.add_point(1,vcolor);

  std::shared_ptr<dray::Points> surface
    = std::make_shared<dray::Points>(col);

  dray::Vec<float,4> vcolor4 = {{color[0],color[1],color[2], 1.f}};
  surface->constant_color(vcolor4);

  surface->field("default");
  surface->color_map().color_table(quad_color_table);
  return surface;
}

TEST (dray_transparency, dray_simple)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "simple_transparency");
  remove_test_image (output_file);

  // Camera
  const int c_width  = 512;
  const int c_height = 512;
  int32 samples = 200;

  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";


  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.set_pos({{0.5f, 0.2f, 18.f}});
  camera.set_look_at({{0.5f, 0.5f, 0.0f}});
  camera.azimuth(0);
  camera.elevate(-30);

  dray::TriangleLight light;
  light.m_v0 = {{4.0f, 0.0f, 8.f}};
  light.m_v1 = {{2.5f, 0.0f, 8.f}};
  light.m_v2 = {{2.5f, 2.0f, 8.f}};

  light.m_intensity[0] = 20.75;
  light.m_intensity[1] = 20.75;
  light.m_intensity[2] = 20.75;

  dray::Material diffuse;
  diffuse.m_specular = 0.f;
  diffuse.m_roughness = 1.f;

  dray::Material trans;
  trans.m_specular = 1.0f;
  trans.m_ior = 1.3f;
  trans.m_spec_trans = 1.0f;
  trans.m_subsurface = 0.5f;
  trans.m_clearcoat = 0.0f;
  trans.m_roughness = .2f;
  trans.m_metallic = 0.4f;

  dray::Material spec;
  spec.m_specular = 1.0f;
  spec.m_ior = 2.3f;
  spec.m_spec_trans = 0.0f;
  spec.m_subsurface = 0.5f;
  spec.m_clearcoat = 0.0f;
  spec.m_roughness = .01f;
  spec.m_metallic = 1.0f;

  dray::TestRenderer renderer;

  float blue[3] = {0.776f, 0.886f, 0.89f};
  float orange[3] = {1.0, 0.388f, 0.35f};
  float grey[3] = {1.0f,1.0f,1.0f};

  // create a quad
  std::vector<double> x = {0.f, 1.f, 1.f, 0.f};
  std::vector<double> y = {0.f, 0.f, 1.f, 1.f};
  float h = 2;
  std::vector<double> z = {h, h, h, h};
  //renderer.add(create_quad(x,y,z,blue,"q1"),trans);
  //renderer.add(create_quad(x,y,z,orange,"q1"),trans);

  std::vector<double> sx = {0.f};
  std::vector<double> sy = {0.f};
  float sh = 1;
  std::vector<double> sz = {sh};
  std::vector<double> sr = {1.f};
  //renderer.add(create_spheres(sx,sy,sz,sr, blue,"q1"),diffuse);
  renderer.add(create_spheres(sx,sy,sz,sr, blue,"q1"),trans);
  //renderer.add(create_spheres(sx,sy,sz,sr, grey,"q1"),spec);

  // create a quad
  //float grey[3] = {.9f,.9f,.9f};
  std::vector<double> xf = {-4.f,  4.f,  4.f, -4.f};
  std::vector<double> yf = {-4.f, -4.f,  4.f, 4.f};
  std::vector<double> zf = {0.f,   0.f,  0.f, 0.f};
  std::shared_ptr<dray::Surface> floor = create_quad(xf,yf,zf,grey,"q2");
  floor->draw_mesh(true);
  floor->mesh_sub_res(20);
  floor->line_thickness(0.6);
  renderer.add(floor,diffuse);

  renderer.samples(samples);
  //renderer.add_light(light);
  renderer.load_env_map(image_file);

  dray::Framebuffer fb = renderer.render(camera);
  fb.background_color({{0.f,0.f,0.f}});
  fb.composite_background();

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file));
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}


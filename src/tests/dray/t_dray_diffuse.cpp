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

dray::Collection create_box(dray::AABB<3> bounds)
{
  conduit::Node dataset;

  const float scale = 6.f;
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
  light.m_intensity[0] = 120.75;
  light.m_intensity[1] = 120.75;
  light.m_intensity[2] = 120.75;
  return light;
}

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

  dray::ColorTable box_color_table;
  box_color_table.clear();
  dray::Vec<float,3> grey = {{.5f,.5f,.5f}};
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

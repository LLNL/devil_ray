// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/filters/mesh_lines.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/png_encoder.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <iostream>
#include <random>


int main (int argc, char *argv[])
{
  std::string config_file = "";

  if (argc != 2)
  {
    std::cout << "Missing configure file name\n";
    exit (1);
  }

  config_file = argv[1];

  Config config (config_file);
  config.load_data ();
  config.load_camera ();
  config.load_field ();


  using SMeshElemT = dray::MeshElem<2u, dray::ElemType::Quad, dray::Order::General>;

  dray::DataSet<SMeshElemT> sdataset =
  dray::MeshBoundary ().template execute<MeshElemT> (config.m_dataset);


  int trials = 5;
  // parse any custon info out of config
  if (config.m_config.has_path ("trials"))
  {
    trials = config.m_config["trials"].to_int32 ();
  }

  dray::Array<dray::Ray> rays;
  config.m_camera.create_rays (rays);

  dray::MeshLines mesh_lines;
  mesh_lines.set_field (config.m_field);

  dray::Framebuffer framebuffer (config.m_camera.get_width (),
                                 config.m_camera.get_height ());

  for (int i = 0; i < trials; ++i)
  {
    framebuffer.clear ();
    config.m_camera.create_rays (rays);
    mesh_lines.execute (rays, sdataset, framebuffer);
  }

  framebuffer.save ("surface_intersection");

  dray::stats::StatStore::write_ray_stats (config.m_camera.get_width (),
                                           config.m_camera.get_height ());
}

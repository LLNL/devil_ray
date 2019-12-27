// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/volume.hpp>
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

  int trials = 5;
  // parse any custon info out of config
  if (config.m_config.has_path ("trials"))
  {
    trials = config.m_config["trials"].to_int32 ();
  }

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.2f);
  color_table.add_alpha (0.1f, 0.2f);
  color_table.add_alpha (0.2f, 0.2f);
  color_table.add_alpha (0.3f, 0.2f);
  color_table.add_alpha (0.4f, 0.2f);
  color_table.add_alpha (0.5f, 0.2f);
  color_table.add_alpha (0.6f, 0.2f);
  color_table.add_alpha (0.7f, 0.2f);
  color_table.add_alpha (0.8f, 0.2f);
  color_table.add_alpha (0.9f, 0.2f);
  color_table.add_alpha (1.0f, 0.1f);

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(config.m_dataset);
  volume->field(config.m_field);
  volume->color_map().color_table(color_table);

  dray::Framebuffer framebuffer;

  dray::Renderer renderer;
  renderer.add(volume);

  for (int i = 0; i < trials; ++i)
  {
    framebuffer = renderer.render(config.m_camera);
  }
  framebuffer.composite_background();
  framebuffer.save ("volume");

  dray::stats::StatStore::write_ray_stats (config.m_camera.get_width (),
                                           config.m_camera.get_height ());
}

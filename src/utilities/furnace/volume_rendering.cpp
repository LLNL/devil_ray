// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
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
  color_table.add_alpha (0.f, 0.0f);
  color_table.add_alpha (0.1f, 0.0f);
  color_table.add_alpha (0.2f, 0.0f);
  color_table.add_alpha (0.3f, 0.2f);
  color_table.add_alpha (0.4f, 0.2f);
  color_table.add_alpha (0.5f, 0.2f);
  color_table.add_alpha (0.6f, 0.2f);
  color_table.add_alpha (0.7f, 0.2f);
  color_table.add_alpha (0.8f, 0.2f);
  color_table.add_alpha (0.9f, 0.2f);
  color_table.add_alpha (1.0f, 0.1f);


  dray::PointLight light;
  light.m_pos= { 0.5f, 0.5f, 0.5f };
  light.m_amb = { 0.5f, 0.5f, 0.5f };
  light.m_diff = { 0.70f, 0.70f, 0.70f };
  light.m_spec = { 0.9f, 0.9f, 0.9f };
  light.m_spec_pow = 90.0;

  dray::Array<dray::PointLight> lights;
  lights.resize(1);
  dray::PointLight *l_ptr = lights.get_host_ptr();
  l_ptr[0] = light;

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(config.m_collection);
  volume->field(config.m_field);
  volume->samples(100);
  volume->color_map().color_table(color_table);

  dray::Array<dray::VolumePartial> partials;

  for (int i = 0; i < trials; ++i)
  {
    dray::Array<dray::Ray> rays;
    config.m_camera.create_rays (rays);
    partials = volume->integrate(rays, lights);

  }

  volume->save("partials",
               partials,
               config.m_camera.get_width(),
               config.m_camera.get_height());

  dray::stats::StatStore::write_ray_stats (config.m_camera.get_width (),
                                           config.m_camera.get_height ());

}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <dray/queries/intersect.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

std::vector<dray::Vec<double,3>>
discritize(const int phi, const int theta)
{

  std::vector<dray::Vec<double,3>> angles;
  const double pi = 3.141592653589793;
  double phi_inc = 360.0 / double(phi);
  double theta_inc = 180.0 / double(theta);
  for(int p = 0; p < phi; ++p)
  {
    float phi  =  -180.f + phi_inc * p;
    //m_phi_values.push_back(phi);

    for(int t = 0; t < theta; ++t)
    {
      float theta = theta_inc * t;

      const int i = p * theta + t;

      //
      //  spherical coords start (r=1, theta = 0, phi = 0)
      //  (x = 0, y = 0, z = 1)
      //
      double phi_r = phi * pi / 180.0;
      double theta_r = theta * pi / 180.0;

      double x = cos(phi_r) * sin(theta_r);
      double y = sin(phi_r) * sin(theta_r);
      double z = cos(theta_r);

      dray::Vec<double,3> dir = {{x,y,z}};
      angles.push_back(dir);
    } // theta
  } // phi

  return angles;
}
#if 0
TEST (dray_intersect, dray_warbly_faces)
{
  std::string root_file = std::string (DATA_DIR) + "warbly_cube/warbly_cube_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "warbly_faces");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  std::vector<dray::Vec<double,3>> ips;
  ips.resize(1);
  ips[0][0] = .5;
  ips[0][1] = .5;
  ips[0][2] = .5;

  std::vector<dray::Vec<double,3>> dirs = discritize(5,5);

  dray::Array<int> face_ids;
  dray::Array<dray::Vec<dray::Float,3>> res_ips;
  dray::Intersect intersector;
  intersector.execute(dataset, dirs, ips, face_ids, res_ips);


  //dray::stats::StatStore::write_ray_stats (c_width, c_height);
}
#else
TEST (dray_intersect, tg)
{
  std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_000190.root";

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "intersect_tg");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  std::vector<dray::Vec<double,3>> ips;
  ips.resize(1);
  ips[0][0] = .5;
  ips[0][1] = .5;
  ips[0][2] = .5;

  std::vector<dray::Vec<double,3>> dirs = discritize(5,5);

  dray::Array<int> face_ids;
  dray::Array<dray::Vec<dray::Float,3>> res_ips;
  dray::Intersect intersector;
  intersector.execute(dataset, dirs, ips, face_ids, res_ips);

}
#endif

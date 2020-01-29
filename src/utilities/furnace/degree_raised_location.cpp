// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/filters/raise_degree.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <iostream>
#include <sstream>
#include <random>


void run_experiment(dray::DataSet &dataset,
                    dray::Array<dray::Vec<float, 3>> &points,
                    int trials,
                    int raise_count,
                    int total_raise);


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

  int num_points = 1000;
  int trials = 5;
  int raise_by = 2;
  int raises = 1;
  // parse any custon info out of config
  if (config.m_config.has_path ("trials"))
  {
    trials = config.m_config["trials"].to_int32 ();
  }
  if (config.m_config.has_path ("points"))
  {
    num_points = config.m_config["points"].to_int32 ();
  }
  if (config.m_config.has_path ("raises"))
  {
    raises = config.m_config["raises"].to_int32 ();
  }
  if (config.m_config.has_path ("raise_by"))
  {
    raise_by = config.m_config["raise_by"].to_int32 ();
  }

  if (raises < 1)
  {
    std::cout << "Error (raises): Must raise at least once for a valid comparision.\n";
    exit(1);
  }
  if (!(1 <= raise_by && raise_by <= 3))
  {
    std::cout << "Error (raise_by): Raising the degree by " << raise_by << " at once is not supported."
                 " Try increasing the 'raises' config variable instead.\n";
    exit(1);
  }

  /// const int num_elems = config.m_dataset.topology()->cells();

  //TODO Also try just raising degree for bounding box but
  // go back to original degree to evaluate the element.


  dray::Array<dray::Vec<float, 3>> points_buffer;
  points_buffer.resize (num_points);

  // Original dataset, no degree raising.
  dray::DataSet dataset = config.m_dataset;
  run_experiment(dataset, points_buffer, trials, 0, 0);

  dray::RaiseDegreeDG degree_raiser;
  for (int a = 1; a <= raises; ++a)
  {
    // Successively degree-raised dataset.
    dataset = degree_raiser.execute(dataset, raise_by);
    run_experiment(dataset, points_buffer, trials, a, a * raise_by);
  }

}


//
// run_experiment()
//
void run_experiment(dray::DataSet &dataset,
                    dray::Array<dray::Vec<float, 3>> &points,
                    int trials,
                    int raise_count,
                    int total_raise)
{

  dray::AABB<3> bounds = dataset.topology()->bounds();

  // random but deterministic
  int seed = 0;
  std::linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647> rgen{ 0 };
  std::uniform_real_distribution<float> dist_x{ bounds.m_ranges[0].min (),
                                                bounds.m_ranges[0].max () };

  std::uniform_real_distribution<float> dist_y{ bounds.m_ranges[1].min (),
                                                bounds.m_ranges[1].max () };

  std::uniform_real_distribution<float> dist_z{ bounds.m_ranges[2].min (),
                                                bounds.m_ranges[2].max () };

  const size_t num_points = points.size();
  dray::Vec<float, 3> *points_ptr = points.get_host_ptr ();
  for (int i = 0; i < num_points; ++i)
  {
    dray::Vec<float, 3> point;
    point[0] = dist_x (rgen);
    point[1] = dist_y (rgen);
    point[2] = dist_z (rgen);
    points_ptr[i] = point;
  }

  dray::stats::StatStore::clear();

  dray::Array<dray::Location> locations;

  for (int i = 0; i < trials; ++i)
  {
    locations = dataset.topology()->locate (points);
  }

  std::stringstream stats_filename_ss;
  stats_filename_ss << "dr_locate_stats_r" << total_raise;

  std::string stats_filename = stats_filename_ss.str();
#ifdef DRAY_STATS
  std::cout << "Writing to file " << stats_filename << "\n";
#endif
  dray::stats::StatStore::write_point_stats (stats_filename);
}

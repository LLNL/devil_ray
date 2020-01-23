// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
/// #include <dray/utils/appstats.hpp>
#include <dray/filters/raise_degree.hpp>

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

  int samples_per_elem = 10;
  int trials = 5;
  int raise_by = 2;
  // parse any custon info out of config
  if (config.m_config.has_path ("trials"))
  {
    trials = config.m_config["trials"].to_int32 ();
  }
  if (config.m_config.has_path ("samples_per_elem"))
  {
    samples_per_elem = config.m_config["samples_per_elem"].to_int32 ();
  }
  if (config.m_config.has_path ("raise_by"))
  {
    raise_by = config.m_config["raise_by"].to_int32 ();
  }

  if (!(1 <= raise_by && raise_by <= 3))
  {
    std::cout << "Raising the degree by " << raise_by << " is not supported.\n";
    exit(1);
  }

  const int num_elems = config.m_dataset.topology()->cells();

  dray::Array<dray::Location> locations;
  locations.resize(samples_per_elem * num_elems);

  // random but deterministic
  int seed = 0;
  std::linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647> rgen{ 0 };
  std::uniform_real_distribution<float> dist{ 0, 1 };

  dray::Location * loc_ptr = locations.get_host_ptr();

  for (int e = 0; e < num_elems; ++e)
    for (int s = 0; s < samples_per_elem; ++s)
    {
      dray::Location loc;
      loc.m_cell_id = e;
      loc.m_ref_pt[0] = dist(rgen);
      loc.m_ref_pt[1] = dist(rgen);
      loc.m_ref_pt[2] = dist(rgen);
      loc_ptr[e * samples_per_elem + s] = loc;
    }

  dray::Array<dray::Vec<float, 3>> wpoints_lo;
  dray::Array<dray::Vec<float, 3>> wpoints_hi;

  wpoints_lo = config.m_dataset.topology()->eval_location(locations);

  dray::RaiseDegreeDG degree_raiser;
  dray::DataSet dataset_hi = degree_raiser.execute(config.m_dataset, raise_by);

  wpoints_hi = dataset_hi.topology()->eval_location(locations);

  const dray::Vec<float, 3> *wpoints_lo_ptr = wpoints_lo.get_host_ptr_const();
  const dray::Vec<float, 3> *wpoints_hi_ptr = wpoints_hi.get_host_ptr_const();

  float e_max = 0.0f;
  float e_min = dray::infinity<float>();
  double e_2 = 0.0;
  for (int i = 0; i < samples_per_elem * num_elems; ++i)
    for (int d = 0; d < 3; ++d)
    {
      float diff = wpoints_lo_ptr[i][d] - wpoints_hi_ptr[i][d];
      if (diff < 0.0f)
        diff = -diff;
      if (e_max < diff)
        e_max = diff;
      if (e_min > diff)
        e_min = diff;
      e_2 += double(diff)*double(diff);
    }
  e_2 /= (samples_per_elem * num_elems);

  std::cout << "Eval diff min: \t " << e_min << "\n";
  std::cout << "Eval diff max: \t " << e_max << "\n";
  std::cout << "Eval diff2 avg: \t" << e_2 << "\n";
}

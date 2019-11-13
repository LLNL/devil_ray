#include <dray/dray.hpp>
#include <dray/utils/appstats.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <iostream>
#include <random>


int main(int argc, char* argv[])
{
  std::string config_file = "";

  if(argc != 2)
  {
    std::cout<<"Missing configure file name\n";
    exit(1);
  }

  config_file = argv[1];

  Config config(config_file);
  config.load_data();

  int num_points = 1000;
  int trials = 5;
  // parse any custon info out of config
  if(config.m_config.has_path("trials"))
  {
    trials = config.m_config["trials"].to_int32();
  }
  if(config.m_config.has_path("points"))
  {
    num_points = config.m_config["points"].to_int32();
  }

  dray::AABB<3> bounds = config.m_dataset.get_mesh().get_bounds();

  dray::Array<dray::Vec<float,3>> points;
  dray::Array<int> active_mask;
  points.resize(num_points);

  // random but deterministic
  int seed = 0;
  std::linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647> rgen{0};
  std::uniform_real_distribution<float> dist_x{bounds.m_ranges[0].min(),
                                               bounds.m_ranges[0].max()};

  std::uniform_real_distribution<float> dist_y{bounds.m_ranges[1].min(),
                                               bounds.m_ranges[1].max()};

  std::uniform_real_distribution<float> dist_z{bounds.m_ranges[2].min(),
                                               bounds.m_ranges[2].max()};

  active_mask.resize(points.size());
  dray::Vec<float,3> *points_ptr = points.get_host_ptr();
  dray::int32 *active_ptr = active_mask.get_host_ptr();

  for(int i = 0; i < num_points; ++i)
  {
    dray::Vec<float,3> point;
    point[0] = dist_x(rgen);
    point[1] = dist_y(rgen);
    point[2] = dist_z(rgen);
    points_ptr[i] = point;
    active_ptr[i] = i;
  }


  dray::Array<dray::RefPoint<float,3>> rpoints; // reference space locations

  rpoints.resize(points.size());
  const dray::RefPoint<float,3> invalid_refpt{ -1, {-1,-1,-1} };

  for(int i = 0; i < trials; ++i)
  {

    rpoints.memset(invalid_refpt);
    config.m_dataset.get_mesh().locate(active_mask, points, rpoints);
  }

  dray::stats::StatStore::write_point_stats("locate_stats");

}

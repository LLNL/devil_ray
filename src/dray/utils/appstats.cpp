#include <dray/utils/global_share.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/array.hpp>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

namespace dray
{
namespace stats
{

namespace detail
{
void write_ray_data(const int32 width,
                    const int32 height,
                    std::vector<std::pair<int32,MattStats>> &ray_data,
                    std::string file_name)
{
  // create a blank field we can fill ine
  const int32 image_size = width * height;

  std::vector<float32> c_field;
  c_field.resize(image_size);
  std::vector<float32> n_field;
  n_field.resize(image_size);

  std::fill(c_field.begin(), c_field.end(), 0.f);
  std::fill(n_field.begin(), n_field.end(), 0.f);

  for(int i = 0; i < ray_data.size(); ++i)
  {
    auto p = ray_data[i];
    c_field[p.first] = p.second.m_candidates;
    n_field[p.first] = p.second.m_newton_iters;
  }

  std::ofstream file;
  file.open (file_name + ".vtk");
  file<<"# vtk DataFile Version 3.0\n";
  file<<"ray data\n";
  file<<"ASCII\n";
  file<<"DATASET STRUCTURED_POINTS\n";
  file<<"DIMENSIONS "<<width + 1<<" "<<height + 1<<" 1\n";

  file<<"CELL_DATA "<<width * height<<"\n";

  file<<"SCALARS candidates float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < image_size; ++i)
  {
    file<<c_field[i]<<"\n";
  }

  file<<"SCALARS newton_iters float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < image_size; ++i)
  {
    file<<n_field[i]<<"\n";
  }

  file.close();
}
} // namespace detail

std::vector<std::vector<std::pair<int32,MattStats>>> StatStore::m_ray_stats;
std::vector<std::vector<std::pair<Vec<float32,3>,MattStats>>> StatStore::m_point_stats;

template<typename T>
void add_ray_stats_impl(Array<Ray<T>> &rays,
                        Array<MattStats> &stats,
                        std::vector<std::vector<std::pair<int32,MattStats>>> &ray_stats)
{
  const int32 size = rays.size();
  std::vector<std::pair<int32,MattStats>> ray_data;
  ray_data.resize(size);
  Ray<T> *ray_ptr = rays.get_host_ptr();
  MattStats *stat_ptr = stats.get_host_ptr();

  for(int i = 0; i < size; ++i)
  {
    Ray<T> ray = ray_ptr[i];
    MattStats mstat = stat_ptr[i];
    ray_data[i] = std::make_pair(ray.m_pixel_id, mstat);
  }

  ray_stats.push_back(std::move(ray_data));
}

template<typename T>
void add_point_stats_impl(Array<Vec<T,3>> &points,
                        Array<MattStats> &stats,
                        std::vector<std::vector<std::pair<Vec<float32,3>,MattStats>>> &point_stats)
{
  const int32 size = points.size();
  std::vector<std::pair<Vec<float32,3>,MattStats>> point_data;
  point_data.resize(size);
  Vec<T,3> *point_ptr = points.get_host_ptr();
  MattStats *stat_ptr = stats.get_host_ptr();

  for(int i = 0; i < size; ++i)
  {
    Vec<T,3> point_t = point_ptr[i];
    Vec<float32,3> point_f;

    point_f[0] = static_cast<float32>(point_t[0]);
    point_f[1] = static_cast<float32>(point_t[1]);
    point_f[2] = static_cast<float32>(point_t[2]);

    MattStats mstat = stat_ptr[i];
    point_data[i] = std::make_pair(point_f, mstat);
  }

  point_stats.push_back(std::move(point_data));
}

void StatStore::write_point_stats(const std::string name)
{
  const int32 num_layers = m_point_stats.size();
  int32 tot_size = 0;
  for(int32 l = 0; l < num_layers; ++l)
  {
    tot_size += m_point_stats[l].size();
  }

  std::ofstream file;
  file.open (name + ".vtk");
  file<<"# vtk DataFile Version 3.0\n";
  file<<"particles\n";
  file<<"ASCII\n";
  file<<"DATASET UNSTRUCTURED_GRID\n";
  file<<"POINTS "<<tot_size<<" double\n";

  for(int32 l = 0; l < num_layers; ++l)
  {
    const int32 size = m_point_stats[l].size();
    for(int32 i = 0; i < size; ++i)
    {
      auto p = m_point_stats[l][i];
      file<<p.first[0]<<" ";
      file<<p.first[1]<<" ";
      file<<p.first[2]<<"\n";
    }
  }

  file<<"CELLS "<<tot_size<<" "<<tot_size* 2<<"\n";
  for(int i = 0; i < tot_size; ++i)
  {
    file<<"1 "<<i<<"\n";
  }

  file<<"CELL_TYPES "<<tot_size<<"\n";
  for(int i = 0; i < tot_size; ++i)
  {
    file<<"1\n";
  }

  file<<"POINT_DATA "<<tot_size<<"\n";
  file<<"SCALARS candidates float\n";
  file<<"LOOKUP_TABLE default\n";

  for(int32 l = 0; l < num_layers; ++l)
  {
    const int32 size = m_point_stats[l].size();
    for(int32 i = 0; i < size; ++i)
    {
      auto p = m_point_stats[l][i];
      file<<p.second.m_candidates<<"\n";
    }
  }

  file<<"SCALARS newton_iters float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int32 l = 0; l < num_layers; ++l)
  {
    const int32 size = m_point_stats[l].size();
    for(int32 i = 0; i < size; ++i)
    {
      auto p = m_point_stats[l][i];
      file<<p.second.m_newton_iters<<"\n";
    }
  }

  file.close();
  m_point_stats.clear();
}

void
StatStore::clear()
{
  m_point_stats.clear();
  m_ray_stats.clear();
}
void
StatStore::add_ray_stats(Array<Ray<float32>> &rays, Array<MattStats> &stats)
{
  add_ray_stats_impl(rays, stats, m_ray_stats);
}

void
StatStore::add_ray_stats(Array<Ray<float64>> &rays, Array<MattStats> &stats)
{
  add_ray_stats_impl(rays, stats, m_ray_stats);
}

void
StatStore::add_point_stats(Array<Vec<float32,3>> &points, Array<MattStats> &stats)
{
  add_point_stats_impl(points, stats, m_point_stats);
}

void
StatStore::add_point_stats(Array<Vec<float64,3>> &points, Array<MattStats> &stats)
{
  add_point_stats_impl(points, stats, m_point_stats);
}

void
StatStore::write_ray_stats(const int32 width,const int32 height)
{
  const int num_images = m_ray_stats.size();
  for(int i = 0; i < num_images; ++i)
  {
    std::stringstream ss;
    ss<<"ray_data_"<<i;
    detail::write_ray_data(width,
                           height,
                           m_ray_stats[i],
                           ss.str());
  }

  m_ray_stats.clear();
}

std::ostream& operator<<(std::ostream &os, const MattStats &stats)
{
  os << "[" << stats.m_newton_iters <<", "<<stats.m_candidates<<"]";
  return os;
}

std::ostream& operator<<(std::ostream &os, const _AppStatsStruct &stats_struct)
{
  os << "t:" << stats_struct.m_total_tests
     << " h:" << stats_struct.m_total_hits
     << " t_iter:" << stats_struct.m_total_test_iterations
     << " h_iter:" << stats_struct.m_total_hit_iterations;
  return os;
}


GlobalShare<AppStats> global_app_stats;

} // namespace stats
} // namespace dray

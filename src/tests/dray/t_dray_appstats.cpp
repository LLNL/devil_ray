#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"

#include <dray/array_utils.hpp>
#include <dray/camera.hpp>
#include <dray/color_table.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/filters/isosurface.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/global_share.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/io/mfem_reader.hpp>

#include <mfem.hpp>

const int c_width = 1024;
const int c_height = 1024;

template<typename T>
dray::Array<dray::ray32>
setup_rays(dray::DataSet<T> &dataset)
{
  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.reset_to_bounds(dataset.get_mesh().get_bounds());
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);
  return rays;
}

TEST(dray_stats, dray_stats_smoke)
{

  std::shared_ptr<dray::stats::AppStats> app_stats_ptr =
    dray::stats::global_app_stats.get_shared_ptr();

  if (app_stats_ptr->is_enabled())
  {
    printf("App stats is enabled.\n");
  }
  else
  {
    printf("App stats is disabled!\n");
  }
}

#include <iostream>
#include <fstream>
void write_particles(dray::Vec<float,3>* points,
                     dray::stats::_AppStatsStruct* stats,
                     const int num_particles)
{
  std::ofstream file;
  file.open ("particles.vtk");
  file<<"# vtk DataFile Version 3.0\n";
  file<<"particles\n";
  file<<"ASCII\n";
  file<<"DATASET UNSTRUCTURED_GRID\n";
  file<<"POINTS "<<num_particles<<" double\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<points[i][0]<<" ";
    file<<points[i][1]<<" ";
    file<<points[i][2]<<"\n";
  }

  file<<"CELLS "<<num_particles<<" "<<num_particles * 2<<"\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<"1 "<<i<<"\n";
  }

  file<<"CELL_TYPES "<<num_particles<<"\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<"1\n";
  }

  file<<"POINT_DATA "<<num_particles<<"\n";
  file<<"SCALARS total_tests float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<stats[i].m_total_tests<<"\n";
  }

  file<<"SCALARS total_hits float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<stats[i].m_total_hits<<"\n";
  }

  file<<"SCALARS total_test_iter float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<stats[i].m_total_test_iterations<<"\n";
  }

  file<<"SCALARS total_hit_iter float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_particles; ++i)
  {
    file<<stats[i].m_total_hit_iterations<<"\n";
  }

  //file<<"SCALARS mass float\n";
  //file<<"LOOKUP_TABLE default\n";
  //for(int i = 0; i < num_particles; ++i)
  //{
  //  const Particle &p = particles[i];
  //  file<<p.m_mass<<"\n";
  //}

  file.close();
}

TEST(dray_stats, dray_stats_locate)
{
  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";

  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name);

  dray::AABB<> bounds = dataset.get_mesh().get_bounds();
  std::cout<<"Bounds "<<bounds<<"\n";

  int grid_dim = 100;

  dray::Array<dray::Vec<float,3>> query_points;
  dray::Array<dray::RefPoint<float,3>> ref_points;
  query_points.resize(grid_dim * grid_dim * grid_dim);
  ref_points.resize(grid_dim * grid_dim * grid_dim);

  const dray::RefPoint<float,3> invalid_refpt{ -1, {-1,-1,-1} };

  dray::RefPoint<float,3>* ref_ptr =  ref_points.get_host_ptr();

  float x_step = bounds.m_ranges[0].length() / float(grid_dim);
  float y_step = bounds.m_ranges[1].length() / float(grid_dim);
  float z_step = bounds.m_ranges[2].length() / float(grid_dim);

  int idx = 0;
  dray::Vec<float,3> * qp_ptr = query_points.get_host_ptr();
  for(int x = 0; x < grid_dim; ++x)
  {
    float x_coord = bounds.m_ranges[0].min() + x_step * x;
    for(int y = 0; y < grid_dim; ++y)
    {
      float y_coord = bounds.m_ranges[1].min() + y_step * y;
      for(int z = 0; z < grid_dim; ++z)
      {
        float z_coord = bounds.m_ranges[2].min() + z_step * z;
        qp_ptr[idx][0] = x_coord;
        qp_ptr[idx][1] = y_coord;
        qp_ptr[idx][2] = z_coord;
        ref_ptr[idx] = invalid_refpt;
        idx++;
      }
    }
  }

  const int num_elems = dataset.get_mesh().get_num_elem();
  dray::stats::AppStats app_stat;
  app_stat.m_query_stats.resize(query_points.size());
  app_stat.m_elem_stats.resize(num_elems);

  dray::Array<dray::int32> active_points;
  active_points.resize(query_points.size());
  dray::int32 *active_ptr = active_points.get_host_ptr();
  for(int i = 0; i < query_points.size(); ++i)
  {
    active_ptr[i] = i;
  }

  dataset.get_mesh().locate(active_points, query_points, ref_points, app_stat);

  write_particles(query_points.get_host_ptr(),
                  app_stat.m_query_stats.get_host_ptr(),
                  query_points.size());

}


TEST(dray_stats, dray_stats_isosurface)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "iso_stats");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "taylor_green/Laghos";
  int cycle = 457;
  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name, cycle);

  dray::ColorTable color_table("cool2warm");

  const float isoval = 0.09;

  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  dray::Array<dray::ray32> rays = setup_rays(dataset);

  dray::Isosurface isosurface;
  isosurface.set_field("Velocity_x");
  isosurface.set_iso_value(isoval);
  isosurface.set_color_table(color_table);
  color_buffer = isosurface.execute(rays, dataset);

  dray::stats::StatStore::write_ray_stats(c_width, c_height);


  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      c_width,
                      c_height);

  png_encoder.save(output_file + ".png");
}

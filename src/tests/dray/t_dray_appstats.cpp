#include "gtest/gtest.h"
#include "test_config.h"

#include <dray/array_utils.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/global_share.hpp>

#include <mfem.hpp>
#include <mfem/fem/conduitdatacollection.hpp>

TEST(dray_stats, dray_stats_smoke)
{

  std::shared_ptr<dray::stats::AppStats> app_stats_ptr = dray::stats::global_app_stats.get_shared_ptr();

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

  mfem::Mesh *mfem_mesh_ptr;
  mfem::GridFunction *mfem_sol_ptr;

  mfem::ConduitDataCollection dcol(file_name); dcol.SetProtocol("conduit_bin"); dcol.Load();
  mfem_mesh_ptr = dcol.GetMesh();
  mfem_sol_ptr = dcol.GetField("bananas");

  if (mfem_mesh_ptr->NURBSext)
  {
     mfem_mesh_ptr->SetCurvature(2);
  }
  mfem_mesh_ptr->GetNodes();

  int space_P;
  dray::ElTransData<float,3> space_data = dray::import_mesh<float>(*mfem_mesh_ptr, space_P);

  int field_P;
  dray::ElTransData<float,1> field_data = dray::import_grid_function<float,1>(*mfem_sol_ptr, field_P);

  dray::Mesh<float> mesh(space_data, space_P);
  dray::Field<float> field(field_data, field_P);
  dray::MeshField<float> mesh_field(mesh, field);

  dray::AABB<> bounds = mesh_field.get_bounds();
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
  std::cout<<"\n";

  const int num_elems = space_data.get_num_elem();
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

  mesh_field.locate(active_points, query_points, ref_points, app_stat);

  write_particles(query_points.get_host_ptr(),
                  app_stat.m_query_stats.get_host_ptr(),
                  query_points.size());

}



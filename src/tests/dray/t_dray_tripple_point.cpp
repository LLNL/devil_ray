#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/timer.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/mfem2dray.hpp>
#include <mfem/fem/datacollection.hpp>

#include <fstream>
#include <stdlib.h>
#include <sstream>

#include <mfem.hpp>
using namespace mfem;

TEST(dray_mfem_blueprint, dray_mfem_blueprint)
{
  //std::string file_name = std::string(DATA_DIR) + "tripple_point/Laghos";
  //std::cout<<"File name "<<file_name<<"\n";
  //mfem::VisItDataCollection col(file_name);
  //int cycle = 7085;
  //col.Load(cycle);
  
  std::string file_name = std::string(DATA_DIR) + "sedov_blast/Laghos";
  std::cout<<"File name "<<file_name<<"\n";
  mfem::VisItDataCollection col(file_name);
  int cycle = 252;
  col.Load(cycle);
  
  dray::ColorTable color_table("Spectral");
  color_table.add_alpha(0.f,  0.01f);
  color_table.add_alpha(0.1f, 0.09f);
  color_table.add_alpha(0.2f, 0.01f);
  color_table.add_alpha(0.3f, 0.09f);
  color_table.add_alpha(0.4f, 0.01f);
  color_table.add_alpha(0.5f, 0.01f);
  color_table.add_alpha(0.6f, 0.01f);
  color_table.add_alpha(0.7f, 0.09f);
  color_table.add_alpha(0.8f, 0.01f);
  color_table.add_alpha(0.9f, 0.01f);
  color_table.add_alpha(1.0f, 0.0f);
  dray::Shader::set_color_table(color_table);

  
  mfem::Mesh *mesh = col.GetMesh();
  mfem::GridFunction *gf = col.GetField("Density");
  //mfem::GridFunction *vec_field = col.GetField("Velocity");
  std::cout<<"Field FECOll "<<gf->FESpace()->FEColl()->Name()<<"\n";
  //std::cout<<"Field FECOll "<<vec_field->FESpace()->FEColl()->Name()<<"\n";
  std::cout<<"Mesh FECOll "<<mesh->GetNodes()->FESpace()->FEColl()->Name()<<"\n";
  if(mesh->NURBSext)
  {
     mesh->SetCurvature(2);
  }

  int space_P;
  dray::ElTransData<float,3> space_data = dray::import_mesh<float>(*mesh, space_P);

  std::cout << "space_data.m_ctrl_idx ...   ";
  space_data.m_ctrl_idx.summary();
  std::cout << "space_data.m_values ...     ";
  space_data.m_values.summary();

  int field_P;

  // Use a component of the vector field.
  //dray::ElTransData<float,1> field_data = dray::import_vector_field_component<float>(*vec_field, 2, field_P);

  dray::ElTransData<float,1> field_data = dray::import_grid_function<float,1>(*gf, field_P);

  std::cout << "field_data.m_ctrl_idx ...   ";
  field_data.m_ctrl_idx.summary();
  std::cout << "field_data.m_values ...     ";
  field_data.m_values.summary();

  /// //DEBUG
  /// const int * space_ctrl_idx_ptr = space_data.m_ctrl_idx.get_host_ptr_const();
  /// const dray::Vec<float,3> *space_data_ptr = space_data.m_values.get_host_ptr_const();
  /// const int el_dof = 27;
  /// printf("ctrl_idx\n");
  /// for (int ii = 0; ii < space_data.m_ctrl_idx.size(); ii += el_dof)
  /// {
  ///   for (int dof_idx = 0; dof_idx < el_dof; dof_idx++)
  ///   {
  ///     int ctrl_idx = space_ctrl_idx_ptr[ii + dof_idx];
  ///     printf("%d ", ctrl_idx);
  ///   }
  ///   printf("\n");
  /// }

  /// printf("values\n");
  /// for (int jj = 0; jj < space_data.m_values.size(); jj++)
  /// {
  ///   dray::Vec<float,3> val = space_data_ptr[jj];
  ///   printf("%02d  ", jj);
  ///   std::cout << val << std::endl;
  /// }


  dray::MeshField<float> mesh_field(space_data, space_P, field_data, field_P);

  //------- DRAY CODE --------

  // Volume rendering.
  dray::Camera camera;
  //camera.set_width(1024);
  //camera.set_height(1024);
  camera.set_width(500);
  camera.set_height(500);
  camera.reset_to_bounds(mesh_field.get_bounds());


  //dray::Vec<dray::float32,3> pos;
  //pos[0] = 4.0;
  //pos[1] = 3.5;
  //pos[2] = 7.5;
  //camera.set_pos(pos);

  dray::ray32 rays;
  camera.create_rays(rays);

  //
  // Volume rendering
  //

  float sample_dist;
  {
    constexpr int num_samples = 100;
    dray::AABB bounds = mesh_field.get_bounds();
    dray::float32 lx = bounds.m_x.length();
    dray::float32 ly = bounds.m_y.length();
    dray::float32 lz = bounds.m_z.length();
    dray::float32 mag = sqrt(lx*lx + ly*ly + lz*lz);
    sample_dist = mag / dray::float32(num_samples);
  }

  dray::Array<dray::Vec<dray::float32,4>> color_buffer = mesh_field.integrate(rays, sample_dist);


  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
  png_encoder.save("tripple_point.png");

  DRAY_LOG_WRITE("mfem");
}

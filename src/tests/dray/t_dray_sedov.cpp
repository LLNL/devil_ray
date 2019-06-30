#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"

#include <dray/camera.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/timer.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/filters/volume_integrator.hpp>
#include <dray/io/mfem_reader.hpp>

#include <mfem/fem/datacollection.hpp>

#include <fstream>
#include <stdlib.h>
#include <sstream>

#include <mfem.hpp>
using namespace mfem;

TEST(dray_mfem_tripple, dray_mfem_tripple_volume)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "sedov_volume");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "sedov_blast/Laghos";
  int cycle = 252;
  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name, cycle);

  dray::ColorTable color_table("Spectral");
  color_table.add_alpha(0.f,  0.00f);
  color_table.add_alpha(0.1f, 0.00f);
  color_table.add_alpha(0.2f, 0.00f);
  color_table.add_alpha(0.3f, 0.00f);
  color_table.add_alpha(0.4f, 0.00f);
  color_table.add_alpha(0.5f, 0.01f);
  color_table.add_alpha(0.6f, 0.01f);
  color_table.add_alpha(0.7f, 0.01f);
  color_table.add_alpha(0.8f, 0.01f);
  color_table.add_alpha(0.9f, 0.01f);
  color_table.add_alpha(1.0f, 0.01f);


  // Volume rendering.
  dray::Camera camera;
  //camera.set_width(1024);
  //camera.set_height(1024);
  camera.set_width(500);
  camera.set_height(500);


  ///dray::Vec<dray::float32,3> pos;
  ///pos[0] = 4.0;
  ///pos[1] = 3.5;
  ///pos[2] = 7.5;
  ///camera.set_pos(pos);
  camera.reset_to_bounds(dataset.get_mesh().get_bounds());

  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  dray::VolumeIntegrator integrator;
  integrator.set_field("Density");
  integrator.set_color_table(color_table);
  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  color_buffer = integrator.execute(rays, dataset);

  dray::PNGEncoder png_encoder;

  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      camera.get_width(),
                      camera.get_height() );

  png_encoder.save(output_file + ".png");
  EXPECT_TRUE(check_test_image(output_file));

  DRAY_LOG_WRITE("mfem");
}

TEST(dray_mfem_tripple, dray_mfem_tripple_iso)
{
  //std::string file_name = std::string(DATA_DIR) + "tripple_point/Laghos";
  //std::cout<<"File name "<<file_name<<"\n";
  //mfem::VisItDataCollection col(file_name);
  //int cycle = 7085;
  //col.Load(cycle);

  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "sedov_iso");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "sedov_blast/Laghos";
  std::cout<<"File name "<<file_name<<"\n";
  mfem::VisItDataCollection col(file_name);
  int cycle = 252;
  col.Load(cycle);

  dray::ColorTable color_table("Spectral");
  dray::Shader::set_color_table(color_table);


  mfem::Mesh *mesh = col.GetMesh();
  mfem::GridFunction *gf = col.GetField("Velocity");
  std::cout<<"Field FECOll "<<gf->FESpace()->FEColl()->Name()<<"\n";
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
  dray::ElTransData<float,1> field_data = dray::import_vector_field_component<float>(*gf, 0, field_P);

  std::cout << "field_data.m_ctrl_idx ...   ";
  field_data.m_ctrl_idx.summary();
  std::cout << "field_data.m_values ...     ";
  field_data.m_values.summary();

  dray::Mesh<float> dray_mesh(space_data, space_P);
  dray::Field<float> field(field_data, field_P);
  dray::MeshField<float> mesh_field(dray_mesh, field);

  //------- DRAY CODE --------

  // Volume rendering.
  dray::Camera camera;
  //camera.set_width(1024);
  //camera.set_height(1024);
  camera.set_width(500);
  camera.set_height(500);
  camera.reset_to_bounds(mesh_field.get_bounds());

  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  float iso_val = .23f;
   dray::Array<dray::Vec4f> color_buffer = mesh_field.isosurface_gradient(rays, iso_val);


  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
  png_encoder.save(output_file + ".png");
  EXPECT_TRUE(check_test_image(output_file));

  DRAY_LOG_WRITE("mfem");
}

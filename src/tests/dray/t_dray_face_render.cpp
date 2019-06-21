#include "gtest/gtest.h"
#include "test_config.h"

#include "t_utils.hpp"
#include <dray/mfem2dray.hpp>
#include <dray/shaders.hpp>
#include <mfem.hpp>
#include <mfem/fem/conduitdatacollection.hpp>

#include <dray/camera.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/Vis/mesh_lines.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>


// TEST()
//
TEST(dray_volume_render, dray_volume_render_simple)
{
  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "impeller_vr");
  remove_test_image(output_file);

  mfem::Mesh *mfem_mesh_ptr;
  /// mfem::GridFunction *mfem_sol_ptr;

  mfem::ConduitDataCollection dcol(file_name);
  dcol.SetProtocol("conduit_bin");
  dcol.Load();
  mfem_mesh_ptr = dcol.GetMesh();
  /// mfem_sol_ptr = dcol.GetField("bananas");

  if (mfem_mesh_ptr->NURBSext)
  {
     mfem_mesh_ptr->SetCurvature(2);
  }
  mfem_mesh_ptr->GetNodes();

  ///mfem_mesh_ptr->Print();
  // Save data for visit comparison
  //mfem::VisItDataCollection visit_dc("visit_mfem", mfem_mesh_ptr);
  //if (false)
  //{
  //   visit_dc.RegisterField("free_bananas",  mfem_sol_ptr);
  //   visit_dc.SetCycle(0);
  //   visit_dc.SetTime(0.0);
  //   visit_dc.Save();
  //}
  // --- DRAY code --- //

  int space_P;
  dray::ElTransData<float,3> space_data = dray::import_mesh<float>(*mfem_mesh_ptr, space_P);

  std::cout << "space_data.m_ctrl_idx ...   ";
  space_data.m_ctrl_idx.summary();
  std::cout << "space_data.m_values ...     ";
  space_data.m_values.summary();

  /// int field_P;
  /// dray::ElTransData<float,1> field_data = dray::import_grid_function<float,1>(*mfem_sol_ptr, field_P);
  //dray::ElTransData<float,1> field_data = dray::import_grid_function<float,1>(mfem_sol_match, field_P);

  /// std::cout << "field_data.m_ctrl_idx ...   ";
  /// field_data.m_ctrl_idx.summary();
  /// std::cout << "field_data.m_values ...     ";
  /// field_data.m_values.summary();

  dray::Mesh<float> mesh(space_data, space_P);  // Works for now only because of the typedef in grid_function_data.hpp
  /// dray::MeshField<float> mesh_field(space_data, space_P, field_data, field_P);

  // Scene bounding box
  /// dray::AABB<3> scene_bounds = dray::LinearBVHBuilder().construct(mesh.get_aabbs()).m_bounds;  // very indirect way
  dray::AABB<3> scene_bounds = dray::reduce(mesh.get_aabbs());  // more direct way.


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

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;
  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.reset_to_bounds(scene_bounds);
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  //
  // Mesh faces rendering
  //
  dray::Array<dray::Vec<dray::float32,4>> color_buffer = dray::mesh_lines<dray::float32>(rays, mesh);//TODO

  //
  // Volume rendering
  //

  {
    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
    png_encoder.save(output_file + ".png");
    EXPECT_TRUE(check_test_image(output_file));
  }

#if 0
   //
   // Isosurface
   //

  camera.create_rays(rays);

  // Output isosurface, colorized by field spatial gradient magnitude.
  {
    float isovalues[5] = { 0.07, 0.005, 0, -8, -15 };
    const char* filenames[5] = {"isosurface_001.png",
                                "isosurface_+08.png",
                                "isosurface__00.png",
                                "isosurface_-08.png",
                                "isosurface_-15.png"};

    for (int iso_idx = 0; iso_idx < 1; iso_idx++)
    {
      std::cout<<"doing iso_surface "<<iso_idx<<" size "<<rays.size()<<"\n";
      dray::Array<dray::Vec4f> color_buffer = mesh_field.isosurface_gradient(rays, isovalues[iso_idx]);
      std::cout<<"done doing iso_surface "<<"\n";
      dray::PNGEncoder png_encoder;
      png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
      png_encoder.save(filenames[iso_idx]);

      printf("Finished rendering isosurface idx %d\n", iso_idx);
    }
  }
  // --- end DRAY  --- //
#endif
}


// --- MFEM code --- //


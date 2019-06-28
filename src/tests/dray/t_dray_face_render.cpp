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
#include <dray/linear_bvh_builder.hpp>

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
  std::string output_file = conduit::utils::join_file_path(output_path, "impeller_faces");
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

  // --- DRAY code --- //

  int space_P;
  dray::ElTransData<float,3> space_data = dray::import_mesh<float>(*mfem_mesh_ptr, space_P);

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

  dray::BVH scene_bvh = dray::LinearBVHBuilder().construct(mesh.get_aabbs());

  //
  // Mesh faces rendering
  //
  {
    dray::Array<dray::Vec<dray::float32,4>> color_buffer = dray::mesh_lines<dray::float32>(rays, mesh, scene_bvh);

    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
    png_encoder.save(output_file + ".png");
    EXPECT_TRUE(check_test_image(output_file));
  }

}


// --- MFEM code --- //


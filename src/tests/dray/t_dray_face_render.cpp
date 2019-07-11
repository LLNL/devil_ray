#include "gtest/gtest.h"
#include "test_config.h"

#include "t_utils.hpp"
#include <dray/shaders.hpp>
#include <dray/io/mfem_reader.hpp>

#include <dray/camera.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/linear_bvh_builder.hpp>

#include <dray/filters/mesh_lines.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>


// TEST()
//
TEST(dray_faces, dray_impeller_faces)
{
  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "impeller_faces");
  remove_test_image(output_file);

  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name);

  dray::Mesh<float32> mesh = dataset.get_mesh();
  dray::AABB<3> scene_bounds = mesh.get_bounds();  // more direct way.

  dray::ColorTable color_table("Spectral");
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
  {
    dray::Array<dray::Vec<dray::float32,4>> color_buffer;
    dray::MeshLines mesh_lines;
    mesh_lines.set_field("bananas");
    color_buffer = mesh_lines.execute(rays, dataset);

    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                        camera.get_width(),
                        camera.get_height() );

    png_encoder.save(output_file + ".png");
    EXPECT_TRUE(check_test_image(output_file));
  }

}

TEST(dray_faces, dray_warbly_faces)
{
  std::string file_name = std::string(DATA_DIR) + "warbly_cube/warbly_cube";
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "warbly_faces");
  remove_test_image(output_file);

  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name);

  dray::Mesh<float32> mesh = dataset.get_mesh();
  dray::AABB<3> scene_bounds = mesh.get_bounds();  // more direct way.

  dray::ColorTable color_table("Spectral");
  dray::Shader::set_color_table(color_table);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;


  dray::Vec<float32,3> center = {0.500501,0.510185,0.495425};
  dray::Vec<float32,3> v_normal = {-0.706176, 0.324936, -0.629072};
  dray::Vec<float32,3> v_pos= center + v_normal * -1.92;



  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.reset_to_bounds(scene_bounds);
  dray::Vec<float32,3> pos = {-.30501f,1.50185f,2.37722f};
  dray::Vec<float32,3> up = {0.f,-1.f,0.f};
  camera.set_pos(pos);
  //camera.set_pos(v_pos);
  camera.set_up(up);
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  dray::Vec<float32,3> top = {0.500501,1.510185,0.495425};
  dray::Vec<float32,3> mov = top - pos;
  mov.normalize();

  dray::PointLightSource light;
  light.m_pos = pos + mov * 4.f;
  light.m_amb = {0.2f, 0.2f, 0.2f};
  light.m_diff = {0.70f, 0.70f, 0.70f};
  light.m_spec = {0.40f, 0.40f, 0.40f};
  light.m_spec_pow = 90.0;
  dray::Shader::set_light_properties(light);
  //
  // Mesh faces rendering
  //
  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  dray::MeshLines mesh_lines;
  mesh_lines.set_field("bananas");
  color_buffer = mesh_lines.execute(rays, dataset);

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      camera.get_width(),
                      camera.get_height() );

  png_encoder.save(output_file + ".png");
  EXPECT_TRUE(check_test_image(output_file));

}


// --- MFEM code --- //


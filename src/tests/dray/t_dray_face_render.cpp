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


#if 0
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
#endif


TEST(dray_faces, dray_crazy_faces)
{
  std::string file_name = std::string(DATA_DIR) + "crazy-hex-bernstein/crazy-hex-bernstein-unidense";
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "crazy-hex-bernstein-faces");
  remove_test_image(output_file);

  /// # L2_T2_3D_P1 = nonconforming positive trilinear
  /// # That implies 8 nodes per element, representing just the vertices.
  /// # There is a single element.
  /// # Hence we define 8 values.

  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name);

  dray::Mesh<float> mesh = dataset.get_mesh();
  dray::AABB<3> scene_bounds = mesh.get_bounds();  // more direct way.
  /// std::cout << "Scene bounds: " << scene_bounds << "\n";

  dray::ColorTable color_table("Spectral");
  dray::Shader::set_color_table(color_table);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;

  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.set_up(dray::make_vec3f(0,1,0));
  camera.set_look_at(dray::make_vec3f(0,0.1,-1));
  camera.reset_to_bounds(scene_bounds);

  camera.set_pos(camera.get_pos()*0.2);

  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  /// // Add uniform field of zeros.
  /// dray::GridFunctionData<float, 1> zero_field_data;
  /// {
  ///   auto meshdata = mesh.get_dof_data();
  ///   zero_field_data.m_ctrl_idx = meshdata.m_ctrl_idx;
  ///   zero_field_data.m_el_dofs = meshdata.m_el_dofs;
  ///   zero_field_data.m_size_el = meshdata.m_size_el;
  ///   zero_field_data.m_size_ctrl = meshdata.m_size_ctrl;

  ///   const int num_unknowns = mesh.get_dof_data().m_size_ctrl;  // hack, violates encapsulation.
  ///   const dray::Vec<float, 1> scalar_zero = {0.0f};
  ///   std::vector<dray::Vec<float, 1>> zero_array(num_unknowns, scalar_zero);
  ///   zero_field_data.m_values = dray::Array<dray::Vec<float, 1>>(zero_array.data(), num_unknowns);
  /// }
  /// dray::Field<float> zero_field(zero_field_data, 20);
  /// dataset.add_field(zero_field, "zeros");

  //
  // Mesh faces rendering
  //
  {
    dray::Array<dray::Vec<dray::float32,4>> color_buffer;
    dray::MeshLines mesh_lines;
    /// mesh_lines.set_field("zeros");
    mesh_lines.set_field("Density");
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
  /// camera.reset_to_bounds(scene_bounds);
  /// dray::Vec<float32,3> pos = {-.30501f,1.50185f,2.37722f};
  /// dray::Vec<float32,3> up = {0.f,-1.f,0.f};
  /// camera.set_pos(pos);
  dray::Vec<float32,3> pos = {-.30501f,1.50185f,2.37722f};
  dray::Vec<float32,3> look_at = {-0.5f, 1.0, -0.5f};
  camera.set_look_at(look_at);
  dray::Vec<float32,3> up = {0.f,0.f,1.f};
  camera.set_up(up);
  camera.reset_to_bounds(scene_bounds);

  camera.set_pos(camera.get_pos()*0.3);

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


#include "gtest/gtest.h"
#include "test_config.h"

#include "t_utils.hpp"
#include <dray/shaders.hpp>
#include <dray/io/mfem_reader.hpp>

#include <dray/camera.hpp>
#include <dray/utils/color_buffer_utils.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/linear_bvh_builder.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/filters/mesh_lines.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST(dray_faces, dray_read)
{
  std::string file_name = std::string(DATA_DIR) + "warbly_cube/warbly_cube";
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "crazy-hex-bernstein-faces");
  remove_test_image(output_file);

  using MeshElemT = dray::MeshElem<float, 3u, dray::ElemType::Quad, dray::Order::General>;
  dray::DataSet<float, MeshElemT> dataset = dray::MFEMReader::load32(file_name);
}

// TEST()
//
TEST(dray_faces, dray_impeller_faces)
{
  //std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  std::string file_name = std::string(DATA_DIR) + "tripple_point/field_dump.cycle";
  std::string output_path = prepare_output_dir();
  //std::string output_file = conduit::utils::join_file_path(output_path, "impeller_faces");
  std::string output_file = conduit::utils::join_file_path(output_path, "tp_faces");
  remove_test_image(output_file);

  // Should not be part of the interface but oh well.
  using MeshElemT = dray::MeshElem<float, 3u, dray::ElemType::Quad, dray::Order::General>;
  using SMeshElemT = dray::MeshElem<float, 2u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;

  dray::DataSet<float, MeshElemT> dataset = dray::MFEMReader::load32(file_name, 6700);

  dray::Mesh<float, MeshElemT> mesh = dataset.get_mesh();
  dray::AABB<3> scene_bounds = mesh.get_bounds();  // more direct way.

  dray::DataSet<float, SMeshElemT> sdataset = dray::MeshBoundary().
      template execute<float, MeshElemT>(dataset);

  dray::ColorTable color_table("Spectral");
  dray::Shader::set_color_table(color_table);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;

  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.reset_to_bounds(scene_bounds);

  dray::Vec<dray::float32,3> pos;
  pos = camera.get_look_at() - dray::make_vec3f(-0.867576, -0.226476, -0.442743) * -8.18535f;
  camera.set_up(dray::make_vec3f(-0.228109, 0.972331, -0.0503851));
  camera.set_pos(pos);

  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  //
  // Mesh faces rendering
  //
  {
    dray::Array<dray::Vec<dray::float32,4>> color_buffer;
    dray::MeshLines mesh_lines;

    //mesh_lines.set_field("bananas");
    mesh_lines.set_field("density");
    color_buffer = mesh_lines.template execute<float, SMeshElemT>(rays, sdataset);

    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                        camera.get_width(),
                        camera.get_height() );

    png_encoder.save(output_file + ".png");
    EXPECT_TRUE(check_test_image(output_file));
  }
  dray::save_depth(rays, c_width, c_height);
#ifdef DRAY_STATS
  dray::stats::StatStore::write_ray_stats(c_width, c_height);
#endif
}


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

  // Should not be part of the interface but oh well.
  using MeshElemT = dray::MeshElem<float, 3u, dray::ElemType::Quad, dray::Order::General>;
  using SMeshElemT = dray::MeshElem<float, 2u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;

  dray::DataSet<float, MeshElemT> dataset = dray::MFEMReader::load32(file_name);

  dray::Mesh<float32, MeshElemT> mesh = dataset.get_mesh();
  dray::AABB<3> scene_bounds = mesh.get_bounds();  // more direct way.

  dray::DataSet<float, SMeshElemT> sdataset = dray::MeshBoundary().
      template execute<float, MeshElemT>(dataset);

  dray::ColorTable color_table("Spectral");
  dray::Shader::set_color_table(color_table);

  // Camera
   const int c_width = 1024;
   const int c_height = 1024;
  //const int c_width = 256;
  //const int c_height = 256;

  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.set_up(dray::make_vec3f(0,1,0));
  camera.set_look_at(dray::make_vec3f(0.0f, 0.0f, 0.0f));
  camera.set_pos(dray::make_vec3f(0.0f, -1.0f, 32.0f));
  /// camera.reset_to_bounds(scene_bounds);
  camera.elevate(-35);
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
  // Depth map
  //
  {
    save_depth(rays, camera.get_width(), camera.get_height(), output_file + "_depth");
  }

  //
  // Mesh faces rendering
  //
  {
    dray::Array<dray::Vec<dray::float32,4>> color_buffer;
    dray::MeshLines mesh_lines;
    mesh_lines.set_field("Density");
    color_buffer = mesh_lines.template execute<float, SMeshElemT>(rays, sdataset);

    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                        camera.get_width(),
                        camera.get_height() );

    png_encoder.save(output_file + ".png");
    EXPECT_TRUE(check_test_image(output_file));
  }

#ifdef DRAY_STATS
  dray::stats::StatStore::write_ray_stats(c_width, c_height);
  dray::stats::StatStore::write_point_stats("impeller_face_stats");
#endif

}


TEST(dray_faces, dray_warbly_faces)
{
  std::string file_name = std::string(DATA_DIR) + "warbly_cube/warbly_cube";
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "warbly_faces");
  remove_test_image(output_file);

  // Should not be part of the interface but oh well.
  using MeshElemT = dray::MeshElem<float, 3u, dray::ElemType::Quad, dray::Order::General>;
  using SMeshElemT = dray::MeshElem<float, 2u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;

  dray::DataSet<float, MeshElemT> dataset = dray::MFEMReader::load32(file_name);

  dray::Mesh<float32, MeshElemT> mesh = dataset.get_mesh();
  dray::AABB<3> scene_bounds = mesh.get_bounds();  // more direct way.

  dray::DataSet<float, SMeshElemT> sdataset = dray::MeshBoundary().
      template execute<float, MeshElemT>(dataset);

  dray::ColorTable color_table("Spectral");

  color_table.clear();
  dray::Vec<float,3> white{1.f, 1.f, 1.f};
  color_table.add_point(0.f, white);
  color_table.add_point(1.f, white);

  dray::Shader::set_color_table(color_table);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;

  //const int c_width = 300;
  //const int c_height = 300;

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
  std::cout<<camera.print();
  //camera.set_pos(v_pos);

  camera.set_pos(camera.get_pos()*0.2);
  camera.elevate(15);
  camera.azimuth(10);


  dray::Vec<float32,3> top = {0.500501,1.510185,0.495425};
  dray::Vec<float32,3> mov = top - camera.get_pos();
  mov.normalize();

  dray::PointLightSource light;
  light.m_pos = pos + mov * 3.f;
  light.m_amb = {0.5f, 0.5f, 0.5f};
  light.m_diff = {0.70f, 0.70f, 0.70f};
  light.m_spec = {0.9f, 0.9f, 0.9f};
  light.m_spec_pow = 90.0;
  dray::Shader::set_light_properties(light);

  //
  // Mesh faces rendering
  //
  dray::Array<dray::Vec<dray::float32,4>> acc;
  acc.resize(camera.get_width() * camera.get_height());
  dray::init_constant(acc, 0.f);

  const int samples = 10;
  dray::Array<dray::ray32> rays;
  for(int i = 0; i < samples; ++i)
  {
    rays.resize(0);
    dray::Array<dray::Vec<dray::float32,4>> color_buffer;
    camera.create_rays_jitter(rays, scene_bounds);

    dray::MeshLines mesh_lines;
    mesh_lines.set_field("bananas");
    color_buffer = mesh_lines.template execute<float, SMeshElemT>(rays, sdataset);

    dray::add(acc, color_buffer);
    std::cout<<"Sample " << i+1 << " of " << samples << "\n";
  }
  dray::scalar_divide(acc, (float)samples);

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) acc.get_host_ptr(),
                      camera.get_width(),
                      camera.get_height() );

  png_encoder.save(output_file + ".png");
  EXPECT_TRUE(check_test_image(output_file));

  //
  // Depth map
  //
  {
    save_depth(rays, camera.get_width(), camera.get_height(), output_file + "_depth");
  }
#ifdef DRAY_STATS
  dray::stats::StatStore::write_ray_stats(c_width, c_height);
#endif
}

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/mfem_reader.hpp>
#include <dray/shaders.hpp>

#include <dray/camera.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/filters/mesh_lines.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

//
TEST (dray_faces, dray_impeller_faces)
{
  std::string file_name = std::string (DATA_DIR) + "impeller/impeller";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "impeller_faces");
  remove_test_image (output_file);

  // Should not be part of the interface but oh well.
  using MeshElemT = dray::MeshElem<3u, dray::ElemType::Quad, dray::Order::General>;
  using SMeshElemT = dray::MeshElem<2u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;

  dray::DataSet<MeshElemT> dataset = dray::MFEMReader::load (file_name, 0);

  dray::Mesh<MeshElemT> mesh = dataset.get_mesh ();
  dray::AABB<3> scene_bounds = mesh.get_bounds (); // more direct way.

  dray::DataSet<SMeshElemT> sdataset =
  dray::MeshBoundary ().template execute<MeshElemT> (dataset);

  dray::ColorTable color_table ("Spectral");
  dray::Shader::set_color_table (color_table);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (scene_bounds);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  //
  // Mesh faces rendering
  //
  {
    dray::MeshLines mesh_lines;

    mesh_lines.set_field ("bananas");
    mesh_lines.draw_mesh (true);
    mesh_lines.template execute<SMeshElemT> (rays, sdataset, framebuffer);

    framebuffer.save (output_file);
    EXPECT_TRUE (check_test_image (output_file));
  }
  framebuffer.save_depth (output_file + "_depth");
#ifdef DRAY_STATS
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
#endif
}


TEST (dray_faces, dray_warbly_faces)
{
  std::string file_name = std::string (DATA_DIR) + "warbly_cube/warbly_cube";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "warbly_faces");
  remove_test_image (output_file);

  // Should not be part of the interface but oh well.
  using MeshElemT = dray::MeshElem<3u, dray::ElemType::Quad, dray::Order::General>;
  using SMeshElemT = dray::MeshElem<2u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;

  dray::DataSet<MeshElemT> dataset = dray::MFEMReader::load (file_name);

  dray::Mesh<MeshElemT> mesh = dataset.get_mesh ();
  dray::AABB<3> scene_bounds = mesh.get_bounds (); // more direct way.

  dray::DataSet<SMeshElemT> sdataset =
  dray::MeshBoundary ().template execute<MeshElemT> (dataset);

  dray::ColorTable color_table ("Spectral");

  color_table.clear ();
  dray::Vec<float, 3> white{ 1.f, 1.f, 1.f };
  color_table.add_point (0.f, white);
  color_table.add_point (1.f, white);

  dray::Shader::set_color_table (color_table);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;

  // const int c_width = 300;
  // const int c_height = 300;

  dray::Vec<float32, 3> center = { 0.500501, 0.510185, 0.495425 };
  dray::Vec<float32, 3> v_normal = { -0.706176, 0.324936, -0.629072 };
  dray::Vec<float32, 3> v_pos = center + v_normal * -1.92;


  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);


  dray::Vec<float32, 3> pos = { -.30501f, 1.50185f, 2.37722f };
  dray::Vec<float32, 3> look_at = { -0.5f, 1.0, -0.5f };
  camera.set_look_at (look_at);
  dray::Vec<float32, 3> up = { 0.f, 0.f, 1.f };
  camera.set_up (up);
  camera.reset_to_bounds (scene_bounds);
  std::cout << camera.print ();
  // camera.set_pos(v_pos);


  dray::Vec<float32, 3> top = { 0.500501, 1.510185, 0.495425 };
  dray::Vec<float32, 3> mov = top - camera.get_pos ();
  mov.normalize ();

  dray::PointLightSource light;
  light.m_pos = pos + mov * 3.f;
  light.m_amb = { 0.5f, 0.5f, 0.5f };
  light.m_diff = { 0.70f, 0.70f, 0.70f };
  light.m_spec = { 0.9f, 0.9f, 0.9f };
  light.m_spec_pow = 90.0;
  dray::Shader::set_light_properties (light);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::MeshLines mesh_lines;

  mesh_lines.set_field ("bananas");
  mesh_lines.draw_mesh (true);
  mesh_lines.template execute<SMeshElemT> (rays, sdataset, framebuffer);

  framebuffer.save (output_file);
  EXPECT_TRUE (check_test_image (output_file));
  framebuffer.save_depth (output_file + "_depth");

#ifdef DRAY_STATS
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
#endif
}

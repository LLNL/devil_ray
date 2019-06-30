#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"
#include <dray/camera.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/filters/volume_integrator.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/timer.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/utils/ray_utils.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/mfem2dray.hpp>
#include <mfem/fem/datacollection.hpp>

#include <fstream>
#include <stdlib.h>
#include <sstream>

#include <mfem.hpp>
using namespace mfem;

TEST(dray_taylor_green, dray_taylor_green_volume)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "taylor_green_volume");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "taylor_green/Laghos";

  int cycle = 457;
  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name, cycle);


  dray::ColorTable color_table("ColdAndHot");
  const float alpha_hi = 0.10f;
  const float alpha_lo = 0.0f;
  color_table.add_alpha(0.0000, alpha_hi);
  color_table.add_alpha(0.0357, alpha_lo);
  color_table.add_alpha(0.0714, alpha_hi);
  color_table.add_alpha(0.1071, alpha_lo);
  color_table.add_alpha(0.1429, alpha_hi);
  color_table.add_alpha(0.1786, alpha_lo);
  color_table.add_alpha(0.2143, alpha_hi);
  color_table.add_alpha(0.2500, alpha_lo);
  color_table.add_alpha(0.2857, alpha_hi);
  color_table.add_alpha(0.3214, alpha_lo);
  color_table.add_alpha(0.3571, alpha_hi);
  color_table.add_alpha(0.3929, alpha_lo);
  color_table.add_alpha(0.4286, alpha_hi);
  color_table.add_alpha(0.4643, alpha_lo);
  color_table.add_alpha(0.5000, alpha_hi);
  color_table.add_alpha(0.5357, alpha_lo);
  color_table.add_alpha(0.5714, alpha_hi);
  color_table.add_alpha(0.6071, alpha_lo);
  color_table.add_alpha(0.6429, alpha_hi);
  color_table.add_alpha(0.6786, alpha_lo);
  color_table.add_alpha(0.7143, alpha_hi);
  color_table.add_alpha(0.7500, alpha_lo);
  color_table.add_alpha(0.7857, alpha_hi);
  color_table.add_alpha(0.8214, alpha_lo);
  color_table.add_alpha(0.8571, alpha_hi);
  color_table.add_alpha(0.8929, alpha_lo);
  color_table.add_alpha(0.9286, alpha_hi);
  color_table.add_alpha(0.9643, alpha_lo);
  color_table.add_alpha(1.0000, alpha_hi);

  dray::PointLightSource light;
  //light.m_pos = {6.f, 3.f, 5.f};
  light.m_pos = {1.2f, -0.15f, 0.4f};
  light.m_amb = {0.1f, 0.1f, 0.1f};
  light.m_diff = {0.70f, 0.70f, 0.70f};
  light.m_spec = {0.30f, 0.30f, 0.30f};
  light.m_spec_pow = 90.0;
  dray::Shader::set_light_properties(light);

  // Volume rendering.
  dray::Camera camera;
  camera.set_width(1024);
  camera.set_height(1024);

  dray::Vec<dray::float32,3> pos;
  pos[0] = .88;
  pos[1] = -.34;
  pos[2] = .32;
  pos = pos * 3.f + dray::make_vec3f(.5,.5,.5);
  camera.set_up(dray::make_vec3f(0,0,1));
  camera.set_pos(pos);
  camera.set_look_at(dray::make_vec3f(0.5, 0.5, 0.5));
  //camera.reset_to_bounds(mesh_field.get_bounds());

  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  //
  // Volume rendering
  //

#ifdef DRAY_STATS
  std::shared_ptr<dray::stats::AppStats> app_stats_ptr =
    dray::stats::global_app_stats.get_shared_ptr();
#endif

  dray::VolumeIntegrator integrator;
  integrator.set_field("Velocity_x");
  integrator.set_color_table(color_table);
  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  color_buffer = integrator.execute(rays, dataset);

#ifdef DRAY_STATS
  app_stats_ptr->m_elem_stats.summary();
#endif

  dray::PNGEncoder png_encoder;

  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      camera.get_width(),
                      camera.get_height() );

  png_encoder.save(output_file + ".png");
  EXPECT_TRUE(check_test_image(output_file));

  DRAY_LOG_WRITE("mfem");
}

TEST(dray_taylor_green, dray_taylor_green_iso)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "taylor_green_iso");
  remove_test_image(output_file);
  remove_test_image(output_file + "_depth");

  std::string file_name = std::string(DATA_DIR) + "taylor_green/Laghos";
  std::cout<<"File name "<<file_name<<"\n";
  mfem::VisItDataCollection col(file_name);
  int cycle = 457;
  col.Load(cycle);

  {
    dray::ColorTable color_table("ColdAndHot");
    dray::Shader::set_color_table(color_table);
  }

  dray::PointLightSource light;
  //light.m_pos = {6.f, 3.f, 5.f};
  light.m_pos = {1.2f, -0.15f, 0.4f};
  light.m_amb = {0.1f, 0.1f, 0.1f};
  light.m_diff = {0.70f, 0.70f, 0.70f};
  light.m_spec = {0.30f, 0.30f, 0.30f};
  light.m_spec_pow = 90.0;
  dray::Shader::set_light_properties(light);

  mfem::Mesh *mesh = col.GetMesh();
  mfem::GridFunction *vec_field = col.GetField("Velocity");

  if(mesh->NURBSext)
  {
     mesh->SetCurvature(2);
  }

  dray::Mesh<float> mesh_data = dray::import_mesh<float>(*mesh);
  dray::Field<float> field_data = dray::import_vector_field_component<float>(*vec_field, 0);

  dray::MeshField<float> mesh_field(mesh_data, field_data);

  std::cerr << "Initialized mesh_field." << std::endl;

  //------- DRAY CODE --------

  // Volume rendering.
  dray::Camera camera;
  camera.set_width(1024);
  camera.set_height(1024);


  dray::Vec<dray::float32,3> pos;
  pos[0] = .88;
  pos[1] = -.34;
  pos[2] = .32;
  pos = pos * 3.f + dray::make_vec3f(.5,.5,.5);
  camera.set_up(dray::make_vec3f(0,0,1));
  camera.set_pos(pos);
  camera.set_look_at(dray::make_vec3f(0.5, 0.5, 0.5));

  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  //
  // Isosurface
  //
  {
    dray::ColorTable color_table("ColdAndHot");
    color_table.add_alpha(0.0000, 1.0f);
    color_table.add_alpha(1.0000, 1.0f);
    dray::Shader::set_color_table(color_table);

#ifdef DRAY_STATS
    std::shared_ptr<dray::stats::AppStats> app_stats_ptr = dray::stats::global_app_stats.get_shared_ptr();
#endif

    //const float isoval = 0.35;
    const float isoval = 0.09;
    dray::Array<dray::Vec4f> iso_color_buffer = mesh_field.isosurface_gradient(rays, isoval);
    printf("done doing iso_surface\n");
    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) iso_color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
    png_encoder.save(output_file + ".png");
    EXPECT_TRUE(check_test_image(output_file));

#ifdef DRAY_STATS
    app_stats_ptr->m_elem_stats.summary();
#endif

    save_depth(rays, camera.get_width(), camera.get_height(), output_file + "_depth");
    EXPECT_TRUE(check_test_image(output_file + "_depth"));

/// #ifdef DRAY_STATS
///     save_wasted_steps(rays, camera.get_width(), camera.get_height(), "wasted_steps_iso.png");
/// #endif
  }


  DRAY_LOG_WRITE("mfem");
}

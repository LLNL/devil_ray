#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
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

TEST(dray_mfem_blueprint, dray_mfem_blueprint)
{
  //std::string file_name = std::string(DATA_DIR) + "tripple_point/Laghos";
  //std::cout<<"File name "<<file_name<<"\n";
  //mfem::VisItDataCollection col(file_name);
  //int cycle = 7085;
  //col.Load(cycle);

  std::string file_name = std::string(DATA_DIR) + "taylor_green/Laghos";
  std::cout<<"File name "<<file_name<<"\n";
  mfem::VisItDataCollection col(file_name);
  int cycle = 457;
  col.Load(cycle);

  {
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
  //mfem::GridFunction *gf = col.GetField("Density");
  mfem::GridFunction *vec_field = col.GetField("Velocity");
  //std::cout<<"Field FECOll "<<gf->FESpace()->FEColl()->Name()<<"\n";
  std::cout<<"Field FECOll "<<vec_field->FESpace()->FEColl()->Name()<<"\n";
  std::cout<<"Mesh FECOll "<<mesh->GetNodes()->FESpace()->FEColl()->Name()<<"\n";
  if(mesh->NURBSext)
  {
     mesh->SetCurvature(2);
  }

  dray::Mesh<float> mesh_data = dray::import_mesh<float>(*mesh);
  dray::Field<float> field_data = dray::import_vector_field_component<float>(*vec_field, 0);
  //dray::Field<float,1> field_data = dray::import_field<float,1>(*gf);

  dray::MeshField<float> mesh_field(mesh_data, field_data);

  std::cerr << "Initialized mesh_field." << std::endl;

  //------- DRAY CODE --------

  // Volume rendering.
  dray::Camera camera;
  camera.set_width(1024);
  camera.set_height(1024);


  ///dray::Vec<dray::float32,3> pos;
  ///pos[0] = 6.46;
  ///pos[1] = 6.23;
  ///pos[2] = 4.41;
  ///pos = pos * .40f;

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

  {
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

#ifdef DRAY_STATS
    std::shared_ptr<dray::AppStats> app_stats_ptr = dray::global_app_stats.get_shared_ptr();
#endif

    dray::Array<dray::Vec<dray::float32,4>> color_buffer = mesh_field.integrate(rays, sample_dist);

#ifdef DRAY_STATS
    app_stats_ptr->m_elem_stats.summary();
#endif

    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
    png_encoder.save("taylor_green_vol.png");



/// #ifdef DRAY_STATS
///   save_wasted_steps(rays, camera.get_width(), camera.get_height(), "wasted_steps_vol.png");
/// #endif
  }

  camera.create_rays(rays);

  //
  // Isosurface
  //
  {
    dray::ColorTable color_table("ColdAndHot");
    color_table.add_alpha(0.0000, 1.0f);
    color_table.add_alpha(1.0000, 1.0f);
    dray::Shader::set_color_table(color_table);

    //const float isoval = 0.35;
    const float isoval = 0.09;
    dray::Array<dray::Vec4f> iso_color_buffer = mesh_field.isosurface_gradient(rays, isoval);
    printf("done doing iso_surface\n");
    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) iso_color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
    char filename[] =  "                                    ";
    sprintf(filename,  "taylor_green_isosurface.png");
    png_encoder.save(filename);

    //DEBUG
    save_depth(rays, camera.get_width(), camera.get_height());

/// #ifdef DRAY_STATS
///     save_wasted_steps(rays, camera.get_width(), camera.get_height(), "wasted_steps_iso.png");
/// #endif
  }



  /// // Oribit path offsets.
  /// constexpr int num_positions = 5;
  /// float angular_pos[num_positions];
  /// {
  ///   const float interp_fac = 1 - expf(-(num_positions-1));
  ///   constexpr float my_pi_over_2 = 1.57079632679;
  ///   float half_opening = my_pi_over_2 * interp_fac;
  ///   angular_pos[0] = half_opening;
  ///   angular_pos[num_positions - 1] = -half_opening;
  ///   for (int ii = 1; ii < num_positions - 1; ii++)
  ///   {
  ///     angular_pos[ii] = angular_pos[0] - ii * 2 * half_opening / (num_positions-1);
  ///   }
  /// }

  /// {
  ///   dray::ColorTable color_table("Spectral");
  ///   color_table.add_alpha(0.f,  1.0f);
  ///   color_table.add_alpha(0.1f, 1.0f);
  ///   color_table.add_alpha(0.2f, 1.0f);
  ///   color_table.add_alpha(0.3f, 1.0f);
  ///   color_table.add_alpha(0.4f, 1.0f);
  ///   color_table.add_alpha(0.5f, 1.0f);
  ///   color_table.add_alpha(0.6f, 1.0f);
  ///   color_table.add_alpha(0.7f, 1.0f);
  ///   color_table.add_alpha(0.8f, 1.0f);
  ///   color_table.add_alpha(0.9f, 1.0f);
  ///   color_table.add_alpha(1.0f, 1.0f);
  ///   dray::Shader::set_color_table(color_table);
  /// }

  /// //for (int ii = 0; ii < num_positions; ii++)
  /// int ii = num_positions / 2;
  /// {
  ///   const float light_radius = 10.0f;
  ///   dray::Shader::set_light_position({light_radius * sinf(angular_pos[ii]), 0, light_radius * cosf(angular_pos[ii])});

  ///   dray::Array<dray::Vec4f> iso_color_buffer = mesh_field.isosurface_gradient(rays, 0.5);
  ///   printf("done doing iso_surface %d\n", ii);
  ///   dray::PNGEncoder png_encoder;
  ///   png_encoder.encode( (float *) iso_color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
  ///   char filename[] =  "                                    ";
  ///   sprintf(filename,  "taylor_green_isosurface%02d.png", ii);
  ///   png_encoder.save(filename);

  ///   //DEBUG
  ///   save_depth(rays, camera.get_width(), camera.get_height());
  /// }

  /// // Many isosurface angles.
  /// {
  ///   const float isoval = 0.5;

  ///   dray::Camera camera;
  ///   camera.set_width(1024);
  ///   camera.set_height(1024);
  ///   camera.set_up(dray::make_vec3f(0,0,1));
  ///   constexpr int num_angles = 6;
  ///   dray::Vec<float,3> cam_pos[num_angles] = {{7,1,8}, {7,1,5}, {7,7,8}, {7,7,5}, {1,7,8}, {1,7,5}};
  ///   for (int cidx = 0; cidx < num_angles; cidx++)
  ///   {
  ///     dray::Vec<float,3> pos = cam_pos[cidx];
  ///     camera.set_pos(pos);
  ///     camera.reset_to_bounds(mesh_field.get_bounds());

  ///     dray::ray32 rays;
  ///     camera.create_rays(rays);

  ///     std::cerr << "Intersecting isosurface " << cidx << "... ";
  ///     mesh_field.intersect_isosurface(rays, isoval);
  ///     save_depth(rays, camera.get_width(), camera.get_height());

  ///     std::cerr << "Done. Type a nonempty string and press [Enter] to continue.  >>";
  ///     std::string unused_string;
  ///     cin >> unused_string;
  ///     std::cerr << "You entered: " << unused_string << std::endl;
  ///   }
  ///   std::cerr << "The loop is over. Exiting." << std::endl;
  /// }


  DRAY_LOG_WRITE("mfem");
}

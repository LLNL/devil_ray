#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/camera.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/shaders.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

#include <dray/filters/isosurface.hpp>
#include <dray/filters/volume_integrator.hpp>

#include <dray/io/mfem_reader.hpp>

#include <mfem/fem/datacollection.hpp>

#include <fstream>
#include <sstream>
#include <stdlib.h>

#include <mfem.hpp>
using namespace mfem;

TEST (dray_mfem_tripple, dray_mfem_tripple_volume)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "sedov_volume");
  remove_test_image (output_file);

  std::string file_name = std::string (DATA_DIR) + "sedov_blast/Laghos";
  int cycle = 252;
  auto dataset = dray::MFEMReader::load (file_name, cycle);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.00f);
  color_table.add_alpha (0.2f, 0.00f);
  color_table.add_alpha (0.3f, 0.00f);
  color_table.add_alpha (0.4f, 0.00f);
  color_table.add_alpha (0.5f, 0.01f);
  color_table.add_alpha (0.6f, 0.01f);
  color_table.add_alpha (0.7f, 0.01f);
  color_table.add_alpha (0.8f, 0.01f);
  color_table.add_alpha (0.9f, 0.01f);
  color_table.add_alpha (1.0f, 0.01f);


  // Volume rendering.
  dray::Camera camera;
  camera.set_width (500);
  camera.set_height (500);

  camera.reset_to_bounds (dataset.get_mesh ().get_bounds ());

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::VolumeIntegrator integrator;
  integrator.set_field ("Density");
  integrator.set_color_table (color_table);

  integrator.execute (rays, dataset, framebuffer);

  framebuffer.save (output_file + ".png");
  EXPECT_TRUE (check_test_image (output_file));

  DRAY_LOG_WRITE ();
}

TEST (dray_mfem_tripple, dray_mfem_tripple_iso)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "sedov_iso");
  remove_test_image (output_file);

  std::string file_name = std::string (DATA_DIR) + "sedov_blast/Laghos";
  int cycle = 252;
  auto dataset = dray::MFEMReader::load (file_name, cycle);

  dray::ColorTable color_table ("Spectral");

  dray::Camera camera;
  camera.set_width (500);
  camera.set_height (500);
  camera.reset_to_bounds (dataset.get_mesh ().get_bounds ());
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  float isoval = .23f;

  dray::Isosurface isosurface;
  isosurface.set_field ("Velocity_x");
  isosurface.set_color_table (color_table);
  isosurface.set_iso_value (isoval);
  isosurface.execute (dataset, rays, framebuffer);

  framebuffer.save (output_file + ".png");
  EXPECT_TRUE (check_test_image (output_file));

  DRAY_LOG_WRITE ();
}

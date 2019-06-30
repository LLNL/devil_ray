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
TEST(dray_volume_render, dray_volume_render_simple)
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
    color_buffer = dray::mesh_lines<dray::float32>(rays, mesh);

    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                        camera.get_width(),
                        camera.get_height() );

    png_encoder.save(output_file + ".png");
    EXPECT_TRUE(check_test_image(output_file));
  }

}


// --- MFEM code --- //


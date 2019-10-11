#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"

#include <dray/camera.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/filters/isosurface.hpp>
/// #include <dray/filters/mesh_lines.hpp>
#include <dray/filters/volume_integrator.hpp>
#include <dray/utils/timer.hpp>
#include <dray/utils/png_encoder.hpp>


const int c_width = 1024;
const int c_height = 1024;

template<class ElemT>
dray::Array<dray::Ray>
setup_rays(dray::DataSet<ElemT> &dataset)
{
  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.reset_to_bounds(dataset.get_mesh().get_bounds());
  dray::Array<dray::Ray> rays;
  camera.create_rays(rays);
  return rays;
}

TEST(dray_performance, dray_performance_volume_test)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "vr_performance");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  using MeshElemT = dray::MeshElem<3u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;
  auto dataset = dray::MFEMReader::load(file_name);

  constexpr int num_trials = 5;

  float elapsed = 0.f;
  dray::Array<dray::Vec<dray::float32,4>> color_buffer;

  for(int i = 0; i < num_trials; ++i)
  {
    dray::Array<dray::Ray> rays = setup_rays<MeshElemT>(dataset);
    dray::Timer timer;

    dray::VolumeIntegrator integrator;
    integrator.set_field("bananas");
    color_buffer = integrator.execute(rays, dataset);

    elapsed += timer.elapsed();
  }

  std::cout<<"Volume Render Ave: "<<elapsed/float(num_trials)<<"\n";

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      c_width,
                      c_height );

  png_encoder.save(output_file + ".png");
}

#if 0
TEST(dray_performance, dray_performance_mesh_lines)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "mesh_performance");
  remove_test_image(output_file);


  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name);

  constexpr int num_trials = 10;

  float elapsed = 0.f;
  dray::Array<dray::Vec<dray::float32,4>> color_buffer;

  for(int i = 0; i < num_trials; ++i)
  {
    dray::Array<dray::ray32> rays = setup_rays(dataset);
    dray::Timer timer;

    dray::MeshLines mesh_lines;
    mesh_lines.set_field("bananas");

    color_buffer = mesh_lines.execute(rays, dataset);

    elapsed += timer.elapsed();
  }

  std::cout<<"Mesh Lines Ave: "<<elapsed/float(num_trials)<<"\n";

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      c_width,
                      c_height );

  png_encoder.save(output_file + ".png");
}
#endif

TEST(dray_performance, dray_isosurface)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "iso_performance");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "taylor_green/Laghos";
  int cycle = 457;
  auto dataset = dray::MFEMReader::load(file_name, cycle);

  dray::ColorTable color_table("cool2warm");

  constexpr int num_trials = 10;

  const float isoval = 0.09;

  float elapsed = 0.f;

  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  for(int i = 0; i < num_trials; ++i)
  {
    dray::Array<dray::Ray> rays = setup_rays(dataset);
    dray::Timer timer;

    dray::Isosurface isosurface;
    isosurface.set_field("Velocity_x");
    isosurface.set_iso_value(isoval);
    isosurface.set_color_table(color_table);
    color_buffer = isosurface.execute(rays, dataset);

    elapsed += timer.elapsed();
  }

  std::cout<<"Isosurface Ave: "<<elapsed/float(num_trials)<<"\n";

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      c_width,
                      c_height);

  png_encoder.save(output_file + ".png");
}

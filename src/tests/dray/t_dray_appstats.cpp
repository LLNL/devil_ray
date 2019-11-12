#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"

#include <dray/array_utils.hpp>
#include <dray/camera.hpp>
#include <dray/color_table.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/filters/isosurface.hpp>
/// #include <dray/filters/mesh_lines.hpp>
#include <dray/filters/slice.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/global_share.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/io/mfem_reader.hpp>

#include <mfem.hpp>

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

void setup_slice_camera(dray::Camera &camera)
{
  camera.set_width(1024);
  camera.set_height(1024);

  dray::Vec<dray::float32,3> pos;
  pos[0] = .5f;
  pos[1] = -1.5f;
  pos[2] = .5f;
  camera.set_up(dray::make_vec3f(0,0,1));
  camera.set_pos(pos);
  camera.set_look_at(dray::make_vec3f(0.5, 0.5, 0.5));
}
#if 0
TEST(dray_stats, dray_stats_isosurface)
{
  dray::stats::StatStore::clear();

  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "iso_stats");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "taylor_green/Laghos";
  int cycle = 457;
  dray::DataSet<float> dataset = dray::MFEMReader::load(file_name, cycle);

  dray::ColorTable color_table("cool2warm");

  const float isoval = 0.09;

  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  dray::Array<dray::Ray> rays = setup_rays(dataset);

  dray::Isosurface isosurface;
  isosurface.set_field("Velocity_x");
  isosurface.set_iso_value(isoval);
  isosurface.set_color_table(color_table);
  color_buffer = isosurface.execute(rays, dataset);

  dray::stats::StatStore::write_ray_stats(c_width, c_height);


  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      c_width,
                      c_height);

  png_encoder.save(output_file + ".png");
}
#endif

TEST(dray_stats, dray_stats_locate)
{
  dray::stats::StatStore::clear();

  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  int cycle = 0;
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "impeller_vr");
  remove_test_image(output_file);

  using MeshElemT = dray::MeshElem<3u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;
  dray::DataSet<MeshElemT> dataset = dray::MFEMReader::load(file_name, cycle);

  dray::AABB<> bounds = dataset.get_mesh().get_bounds();
  std::cout<<"Bounds "<<bounds<<"\n";

  int grid_dim = 20;

  dray::Array<dray::Vec<float,3>> query_points;
  query_points.resize(grid_dim * grid_dim * grid_dim);


  float x_step = bounds.m_ranges[0].length() / float(grid_dim);
  float y_step = bounds.m_ranges[1].length() / float(grid_dim);
  float z_step = bounds.m_ranges[2].length() / float(grid_dim);

  int idx = 0;
  dray::Vec<float,3> * qp_ptr = query_points.get_host_ptr();
  for(int x = 0; x < grid_dim; ++x)
  {
    float x_coord = bounds.m_ranges[0].min() + x_step * x;
    for(int y = 0; y < grid_dim; ++y)
    {
      float y_coord = bounds.m_ranges[1].min() + y_step * y;
      for(int z = 0; z < grid_dim; ++z)
      {
        float z_coord = bounds.m_ranges[2].min() + z_step * z;
        qp_ptr[idx][0] = x_coord;
        qp_ptr[idx][1] = y_coord;
        qp_ptr[idx][2] = z_coord;
        idx++;
      }
    }
  }

  const int num_elems = dataset.get_mesh().get_num_elem();

  dray::Array<dray::Location> locs = dataset.get_mesh().locate(query_points);

  dray::stats::StatStore::write_point_stats("locate_stats");
}

TEST(dray_stats, dray_slice_stats)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "slice_stats");
  remove_test_image(output_file);

  std::string file_name = std::string(DATA_DIR) + "taylor_green/Laghos";

  int cycle = 457;
  using MeshElemT = dray::MeshElem<3u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;
  /// dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name, cycle);
  auto dataset = dray::MFEMReader::load(file_name, cycle);

  dray::Camera camera;
  setup_slice_camera(camera);

  dray::Array<dray::Ray> rays;
  camera.create_rays(rays);
  dray::Framebuffer framebuffer(camera.get_width(), camera.get_height());

  dray::Vec<float,3> point;
  point[0] = 0.5f;
  point[1] = 0.5;
  point[2] = -1.e-5f;

  //dray::Vec<float,3> normal;

  dray::Slice slicer;
  slicer.set_field("Velocity_x");
  slicer.set_point(point);
  slicer.execute(rays, dataset, framebuffer);

  framebuffer.save(output_file);

  dray::stats::StatStore::write_point_stats("slice_stats");
}
#if 0
TEST(dray_appstats, dray_stats_mesh_lines)
{
  dray::stats::StatStore::clear();
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "mesh_stats");
  remove_test_image(output_file);


  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name);

  dray::Array<dray::Vec<dray::float32,4>> color_buffer;

  dray::Array<dray::ray32> rays = setup_rays(dataset);

  dray::MeshLines mesh_lines;

  color_buffer = mesh_lines.execute(rays, dataset);

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      c_width,
                      c_height );

  png_encoder.save(output_file + ".png");
  dray::stats::StatStore::write_ray_stats(c_width, c_height);
}
#endif

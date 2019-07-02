#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"

#include <dray/io/mfem_reader.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/aabb.hpp>
#include <dray/filters/attractor_map.hpp>
#include <dray/GridFunction/grid_function_data.hpp>


/// const int grid_depth = 10;  // 1024x1024
const int grid_depth = 4;
const int c_width  = 1 << grid_depth;
const int c_height = 1 << grid_depth;


TEST(dray_attractors, dray_attractors_2d)
{
  std::string file_name = std::string(DATA_DIR) + "warbly_cube/warbly_cube";
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "warbly_cube_attractors");
  remove_test_image(output_file);

  // Get mesh/cell.
  dray::DataSet<float> dataset = dray::MFEMReader::load32(file_name);

  /// // What coordinates are actually inside the first cell?
  /// {
  ///   dray::AABB<3> cell_bounds;
  ///   dataset.get_mesh().access_host_mesh().get_elem(0).get_bounds(cell_bounds);
  ///   std::cout << "First element bounds: " << cell_bounds << "\n";
  /// }

  const int el_id = 0;  // Use one cell for all queries/guesses.

  // Define query point.
  const dray::Vec<float,3> query_point({0.5, 0.5, 0.5});

  // Define collection of sample initial guesses.
  dray::Array<dray::RefPoint<float,3>> sample_guesses = dray::AttractorMap::domain_grid_slice_xy<float>(grid_depth, grid_depth, 0.5, el_id);

  // Get image.
  dray::AttractorMap attractor_map_filter;
  dray::Array<dray::Vec<dray::float32, 4>> color_buffer = attractor_map_filter.execute<float>(
      query_point,
      sample_guesses,
      dataset);

  // Encode image.
  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(), c_width, c_height);
  png_encoder.save(output_file + ".png");

  // Check against benchmark.
  EXPECT_TRUE(check_test_image(output_file));
}

#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"

#include <dray/io/mfem_reader.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/aabb.hpp>
#include <dray/filters/attractor_map.hpp>
#include <dray/GridFunction/grid_function_data.hpp>


const int grid_depth = 10;  // 1024x1024
const int c_width  = 1 << grid_depth;
const int c_height = 1 << grid_depth;



#include <iostream>
#include <fstream>
void write_attractor_vtk_image(
    const int nx,
    const int ny,
    const int nz,
    const dray::Vec<float,3> *solutions,
    const dray::int32 *iterations)
{
  const int num_cells = nx * ny * nz;

  const double space_x = (nx > 1 ? 1.0/(nx-1) : 1);
  const double space_y = (ny > 1 ? 1.0/(ny-1) : 1);
  const double space_z = (nz > 1 ? 1.0/(nz-1) : 1);

  std::ofstream file;
  file.open ("attractors.vtk");
  file<<"# vtk DataFile Version 3.0\n";
  file<<"attractors\n";
  file<<"ASCII\n";
  file<<"DATASET STRUCTURED_POINTS\n";
  file<<"DIMENSIONS " << nx+1 << " " << ny+1 << " " << nz+1 << "\n";
  file<<"ORIGIN 0 0 0\n";
  file<<"SPACING " << space_x << " " << space_y << " " << space_z << "\n";

  file<<"CELL_DATA "<<num_cells<<"\n";

  file<<"SCALARS iterations int 1\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_cells; ++i)
  {
    file << iterations[i] << "\n";
  }

  file<<"SCALARS edge_dist float 1\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_cells; ++i)
  {
    file << dray::AttractorMapShader::edge_dist<float>(solutions[i]) << "\n";
  }

  file<<"VECTORS attractor float\n";
  file<<"LOOKUP_TABLE default\n";
  for(int i = 0; i < num_cells; ++i)
  {
    file << solutions[i][0] << " " << solutions[i][1] << " " << solutions[i][2] << "\n";
  }

  file.close();
}



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
  const dray::Vec<float,3> query_point({1.0, 1.0, 1.0});

  /// // Query point produced from a point that is definitely inside or on the element.
  /// dray::Vec<float,3> query_point;
  /// dray::Vec<dray::Vec<float,3>,3> unused_deriv;
  /// dataset.get_mesh().access_host_mesh().get_elem(el_id).eval({1.0, 1.0, 1.0}, query_point, unused_deriv);

  // Define collection of sample initial guesses.
  const dray::Array<dray::RefPoint<float,3>> sample_guesses = dray::AttractorMap::domain_grid_slice_xy<float>(grid_depth, grid_depth, 0.5, el_id);

  // Other outputs (for vtk file).
  dray::Array<dray::Vec<float,3>> solutions;
  dray::Array<int> iterations;

  // Get image.
  dray::AttractorMap attractor_map_filter;
  dray::Array<dray::Vec<dray::float32, 4>> color_buffer = attractor_map_filter.execute<float>(
      query_point,
      sample_guesses,
      solutions,
      iterations,
      dataset);

  // Encode image.
  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(), c_width, c_height);
  png_encoder.save(output_file + ".png");

  /// // If one or more solutions were found inside the element, what was the first one?
  /// const int solutions_size = solutions.size();
  /// const dray::Vec<float,3> *solutions_ptr = solutions.get_host_ptr();
  /// int sidx;
  /// for (sidx = 0; sidx < solutions_size; sidx++)
  /// {
  ///   if (dray::MeshElem<float,3>::is_inside(solutions_ptr[sidx]))
  ///     break;
  /// }
  /// if (sidx < solutions_size)
  ///   std::cout << "Solution found: sidx==" << sidx << " \t ref_coords==" << solutions_ptr[sidx] << "\n";

  // Dump VTK file of (solutions, iterations).
  write_attractor_vtk_image(
      c_width,
      c_height,
      1,                                   // 2D slab image.
      solutions.get_host_ptr_const(),
      iterations.get_host_ptr_const());

  // Check against benchmark.
  EXPECT_TRUE(check_test_image(output_file));
}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/mfem_reader.hpp>
#include <dray/shaders.hpp>

#include <dray/camera.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/filters/mesh_boundary.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>


// TEST()
//
TEST (dray_mesh_boundary, dray_impeller_boundary)
{
  std::string file_name = std::string (DATA_DIR) + "impeller/impeller";
  /// std::string output_path = prepare_output_dir();
  /// std::string output_file = conduit::utils::join_file_path(output_path,
  /// "impeller_faces"); remove_test_image(output_file);

  auto dataset = dray::MFEMReader::load32 (file_name);

  auto mesh = dataset.get_mesh ();
  dray::AABB<3> scene_bounds = mesh.get_bounds (); // more direct way.

  /// dray::ColorTable color_table("Spectral");
  /// dray::Shader::set_color_table(color_table);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (scene_bounds);
  dray::Array<dray::ray32> rays;
  camera.create_rays (rays);

  //
  // Extract the boundary surface mesh.
  //
  dray::MeshBoundary mesh_boundary;
  auto dataset_2d = mesh_boundary.execute (dataset);

  std::cout << "dataset_2d.get_mesh().get_poly_order()=="
            << dataset_2d.get_mesh ().get_poly_order () << "\n";

  dray::AABB<3> scene_bounds_2d = dataset_2d.get_mesh ().get_bounds (); // bounds of surface.

  std::cout << "3D bounds: " << scene_bounds << "\n";
  std::cout << "2D bounds: " << scene_bounds_2d << "\n";
  std::cout << "Contained ? "
            << (int)scene_bounds_2d.is_contained_in (scene_bounds) << "\n";

  //
  // Rendering
  //
  {
  }
}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/line_renderer.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace dray;

void generate_lines(
  Array<Vec<float32,3>> &starts, 
  Array<Vec<float32,3>> &ends, 
  int num_lines,
  const int width,
  const int height)
{
  starts.resize(num_lines);
  ends.resize(num_lines);

  Vec<float32,3> *starts_ptr = starts.get_host_ptr();
  Vec<float32,3> *ends_ptr = ends.get_host_ptr();

  for (int i = 0; i < num_lines; i ++)
  {

    int x1 = rand() % width;
    int y1 = rand() % height;
    int x2 = rand() % width;
    int y2 = rand() % height;

    starts_ptr[i] = {{(float) x1, (float) y1, 0.f}};
    ends_ptr[i] = {{(float) x2, (float) y2, 0.f}};
  }
}

Vec<float32,3> xyz(Vec<float32,4> v)
{
  return {{v[0], v[1], v[2]}};
}

// TEST (dray_faces, dray_impeller_faces)
// {
//   std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
//   std::string output_path = prepare_output_dir ();
//   std::string output_file = "hereiam";
//   // conduit::utils::join_file_path (output_path, "lines_test");
//   // remove_test_image (output_file);

//   Collection dataset = BlueprintReader::load (root_file);

//   ColorTable color_table ("Spectral");

//   // Camera
//   const int c_width  = 1024;
//   const int c_height = 1024;

//   Camera camera;
//   camera.set_width (c_width);
//   camera.set_height (c_height);
//   camera.reset_to_bounds (dataset.bounds());

//   AABB<3> aabb = dataset.bounds();

//   srand(time(NULL));

//   for (int i = 0; i < 5; i ++)
//   {
//     int num_lines = 1000;
//     Array<Vec<float32,3>> starts;
//     Array<Vec<float32,3>> ends;
//     Matrix<float32, 4, 4> transform;
//     transform.identity();
//     generate_lines(starts, ends, num_lines, c_width, c_height);

//     dray::Framebuffer fb1;
//     dray::Framebuffer fb2;
//     LineRenderer lines;

//     lines.render(fb1, transform, starts, ends);
//     lines.justinrender(fb2, transform, starts, ends);

//     std::cout << "==========" << std::endl;

//     fb1.composite_background();
//     fb2.composite_background();
    
//     fb1.save(output_file + "1");
//     fb2.save(output_file + "2");
//     // fb.save_depth (output_file + "_depth");
//   }
// }

TEST (dray_faces, dray_aabb)
{
  std::string root_file = std::string (DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file = "hereiam";
  // conduit::utils::join_file_path (output_path, "lines_test");
  // remove_test_image (output_file);

  Collection dataset = BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(dataset);

  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  ColorTable color_table ("Spectral");

  std::shared_ptr<dray::Surface> surface
      = std::make_shared<dray::Surface>(faces);
  surface->field("diffusion");
  surface->color_map().color_table(color_table);
  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);

  AABB<3> aabb = dataset.bounds();

  int num_lines = 12;
  Array<Vec<float32,3>> starts;
  Array<Vec<float32,3>> ends;
  starts.resize(num_lines);
  ends.resize(num_lines);

  Matrix<float32, 4, 4> transform;
  Matrix<float32, 4, 4> V = camera.view_matrix();

  LineRenderer lines;

  Vec<float32,3> *starts_ptr = starts.get_host_ptr();
  Vec<float32,3> *ends_ptr = ends.get_host_ptr();

  float minx, miny, minz, maxx, maxy, maxz;

  minx = aabb.m_ranges[0].min();
  miny = aabb.m_ranges[1].min();
  minz = aabb.m_ranges[2].min();
  maxx = aabb.m_ranges[0].max();
  maxy = aabb.m_ranges[1].max();
  maxz = aabb.m_ranges[2].max();

  Vec<float32,4> o,i,j,k,ij,ik,jk,ijk;
  o = V * ((Vec<float32,4>) {{minx, miny, minz, 1.f}});
  i = V * ((Vec<float32,4>) {{maxx, miny, minz, 1.f}});
  j = V * ((Vec<float32,4>) {{minx, maxy, minz, 1.f}});
  k = V * ((Vec<float32,4>) {{minx, miny, maxz, 1.f}});
  ij = V * ((Vec<float32,4>) {{maxx, maxy, minz, 1.f}});
  ik = V * ((Vec<float32,4>) {{maxx, miny, maxz, 1.f}});
  jk = V * ((Vec<float32,4>) {{minx, maxy, maxz, 1.f}});
  ijk = V * ((Vec<float32,4>) {{maxx, maxy, maxz, 1.f}});

  float near, far;
  float z_values[] = {o[2], i[2], j[2], k[2], ij[2], ik[2], jk[2], ijk[2]};
  near = z_values[0];
  far = z_values[0];
  for (int i = 1; i < 8; i ++)
  {
    if (z_values[i] < near)
    {
      near = z_values[i];
    }
    if (z_values[i] > far)
    {
      far = z_values[i];
    }
  }

  near = abs(near);
  far = abs(far);

  if (near > far)
  {
    float temp = far;
    far = near;
    near = temp;
  }
  
  Matrix<float32, 4, 4> P = camera.projection_matrix(near - 1.f, far + 1.f);

  transform = P;

  starts_ptr[0] = xyz(o);
  ends_ptr[0] = xyz(i);
  starts_ptr[1] = xyz(o);
  ends_ptr[1] = xyz(j);
  starts_ptr[2] = xyz(o);
  ends_ptr[2] = xyz(k);
  starts_ptr[3] = xyz(i);
  ends_ptr[3] = xyz(ik);
  starts_ptr[4] = xyz(i);
  ends_ptr[4] = xyz(ij);
  starts_ptr[5] = xyz(j);
  ends_ptr[5] = xyz(jk);
  starts_ptr[6] = xyz(j);
  ends_ptr[6] = xyz(ij);
  starts_ptr[7] = xyz(ij);
  ends_ptr[7] = xyz(ijk);
  starts_ptr[8] = xyz(k);
  ends_ptr[8] = xyz(ik);
  starts_ptr[9] = xyz(k);
  ends_ptr[9] = xyz(jk);
  starts_ptr[10] = xyz(ik);
  ends_ptr[10] = xyz(ijk);
  starts_ptr[11] = xyz(jk);
  ends_ptr[11] = xyz(ijk);

  lines.render(fb, transform, starts, ends);

  fb.composite_background();

  fb.save(output_file + "aabb");
  fb.save_depth (output_file + "_depth");
}


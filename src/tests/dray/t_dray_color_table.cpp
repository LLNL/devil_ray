// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"
#include <dray/color_table.hpp>
#include <dray/utils/png_encoder.hpp>

using namespace dray;

void write_color_table (const std::string name, const std::string file_name)
{
  dray::ColorTable color_table (name);

  const int samples = 1024;
  Array<Vec<float32, 4>> color_map;
  color_table.sample (samples, color_map);


  const int width = 1024;
  const int height = 100;
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize (width * height);

  const Vec<float32, 4> *color_ptr = color_map.get_host_ptr ();
  Vec<float32, 4> *buffer_ptr = color_buffer.get_host_ptr ();

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      float32 t = float32 (x) / float32 (width);
      int32 color_idx = static_cast<int32> (t * (float32 (samples) - 1.f));
      int32 buffer_idx = y * width + x;
      buffer_ptr[buffer_idx] = color_ptr[color_idx];
    }
  }

  PNGEncoder encoder;
  const float32 *start = &(buffer_ptr[0][0]);
  encoder.encode (start, width, height);
  encoder.save (file_name + ".png");
}

TEST (dray_test, dray_color_table)
{
  std::string output_path = prepare_output_dir ();

  std::vector<std::string> color_tables;
  color_tables.push_back ("cool2warm");
  // color_tables.push_back("grey");
  // color_tables.push_back("blue");
  // color_tables.push_back("orange");
  // color_tables.push_back("temperature");
  // color_tables.push_back("rainbow");
  // color_tables.push_back("levels");
  // color_tables.push_back("thermal");
  // color_tables.push_back("IsoL");
  // color_tables.push_back("CubicL");
  // color_tables.push_back("CubicYF");
  // color_tables.push_back("LinearL");
  // color_tables.push_back("LinLhot");
  // color_tables.push_back("PuRd");
  // color_tables.push_back("Accent");
  // color_tables.push_back("Blues");
  // color_tables.push_back("BrBG");
  // color_tables.push_back("BuGn");
  // color_tables.push_back("BuPu");
  // color_tables.push_back("Dark2");
  // color_tables.push_back("GnBu");
  // color_tables.push_back("Greens");
  // color_tables.push_back("Greys");
  // color_tables.push_back("Oranges");
  // color_tables.push_back("OrRd");
  // color_tables.push_back("Paired");
  // color_tables.push_back("Pastel1");
  // color_tables.push_back("Pastel2");
  // color_tables.push_back("PiYG");
  // color_tables.push_back("PRGn");
  // color_tables.push_back("PuBu");
  // color_tables.push_back("PuBuGn");
  // color_tables.push_back("PuOr");
  // color_tables.push_back("PuRd");
  // color_tables.push_back("Purples");
  // color_tables.push_back("RdBu");
  // color_tables.push_back("RdGy");
  // color_tables.push_back("RdPu");
  // color_tables.push_back("RdYlBu");
  // color_tables.push_back("RdYlGn");
  // color_tables.push_back("Reds");
  // color_tables.push_back("Set1");
  // color_tables.push_back("Set2");
  // color_tables.push_back("Set3");
  // color_tables.push_back("Spectral");
  // color_tables.push_back("YlGnBu");
  // color_tables.push_back("YlGn");
  // color_tables.push_back("YlOrBr");
  // color_tables.push_back("YlOrRd");
  for (int i = 0; i < color_tables.size (); ++i)
  {
    std::string output_file =
    conduit::utils::join_file_path (output_path, color_tables[i]);
    remove_test_image (output_file);
    write_color_table (color_tables[i], output_file);
    // check that we created an image
    EXPECT_TRUE (check_test_image (output_file));
  }
}

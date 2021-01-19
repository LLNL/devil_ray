// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/hdr_image_reader.hpp>

#include <dray/rendering/renderer.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_faces, dray_hdr_reader)
{
  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "hdr_image");
  remove_test_image (output_file);

  int width, height;
  dray::Array<dray::Vec<float,3>> image = dray::read_hdr_image(image_file, width, height);
  std::cout<<"Image dims ("<<width<<","<<height<<")\n";
  dray::Framebuffer fb(width,height);
  const int32 size = width * height;
  dray::Vec<float,3> * in_ptr = image.get_host_ptr();
  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();
  for(int i = 0; i < size; ++i)
  {
    dray::Vec<float,3> in_color = in_ptr[i];
    dray::Vec<float,4> out_color = {{in_color[0], in_color[1], in_color[2], 1.f}};;
    out_ptr[i] = out_color;
  }
  fb.tone_map();
  fb.save(output_file);
}

//TEST (dray_faces, dray_hdr_mapping)
//{
//  std::string image_file = std::string (DATA_DIR) + "spiaggia_di_mondello_2k.hdr";
//  std::string output_path = prepare_output_dir ();
//  std::string output_file =
//  conduit::utils::join_file_path (output_path, "hdr_image");
//  remove_test_image (output_file);
//
//  int width, height;
//  dray::Array<dray::Vec<float,3>> image = dray::read_hdr_image(image_file, width, height);
//  std::cout<<"Image dims ("<<width<<","<<height<<")\n";
//  dray::Framebuffer fb(width,height);
//  const int32 size = width * height;
//  dray::Vec<float,3> * in_ptr = image.get_host_ptr();
//  dray::Vec<float,4> * out_ptr = fb.colors().get_host_ptr();
//  for(int i = 0; i < size; ++i)
//  {
//    dray::Vec<float,3> in_color = in_ptr[i];
//    dray::Vec<float,4> out_color = {{in_color[0], in_color[1], in_color[2], 1.f}};;
//    out_ptr[i] = out_color;
//  }
//  fb.tone_map();
//  fb.save(output_file);
//}

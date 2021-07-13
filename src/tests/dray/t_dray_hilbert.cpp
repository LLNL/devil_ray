// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

/// #include <conduit_relay.hpp>
/// #include <conduit_blueprint.hpp>

#include <dray/policies.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

#include <dray/fixed_point.hpp>
#include <dray/lattice_coordinate.hpp>
#include <dray/tree_node.hpp>
#include <dray/sfc.hpp>

#include <dray/utils/png_decoder.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/device_framebuffer.hpp>

#include <RAJA/RAJA.hpp>

#include <array>
#include <cmath>
#include <bitset>


float color_mag_to_float(unsigned char a)
{
  return a / 255.f;
}

unsigned char float_to_color_mag(float f)
{
  return 255.f * f;
}

// Convert lower-left-origin (u,v) lookup, convert uchar to floats.
const unsigned char * lookup_rgba(const unsigned char *rgba_ptr,
                                   const int width,
                                   const int height,
                                   const int u,
                                   const int v)
{
  const int inOffset = ((height - v - 1) * width + u) * 4;
  return rgba_ptr + inOffset;
}



TEST (dray_hilbert, dray_continuous_color)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "hilbert_path");
  remove_test_image (output_file);

  using dray::Float;
  using dray::int32;

  const int levels = 4;
  const int side_length = 1u << levels;

  dray::Framebuffer frame_buffer(side_length, side_length);
  frame_buffer.clear({{0, 0, 0, 0}});

  dray::DeviceFramebuffer dvc_frame_buffer(frame_buffer);

  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, side_length * side_length),
      [=, &dvc_frame_buffer] DRAY_LAMBDA (int index)
  {
    const int j = index / side_length,  i = index % side_length;

    dray::TreeNode<2> node = dray::TreeNode<2>::root();
    dray::SFC<2> sfc;
    for (int32 l = 0; l < levels; ++l)
    {
      int32 sub_index = (index >> 2*(levels-1-l)) & 3;
      int32 child_num = sfc.child_num(sub_index);
      node = node.child(child_num);
      sfc = sfc.subcurve(sub_index);
    }
    dray::Vec<Float, 2> v = node.coord().vec<Float>() * side_length;
    dray::Vec<int32, 2> u = {{int32(v[0]), int32(v[1])}};
    /// fprintf(stdout, "(%2d,%2d)\n", u[0], u[1]);

    const Float mag = color_mag_to_float(index % 256);
    dray::Vec<Float, 4> color = {{mag, mag, mag, 1.0f}};
    dvc_frame_buffer.set_color(u[0] + u[1] * side_length, color);
  });

  frame_buffer.save(output_file);

  /// conduit::Node conduit_frame_buffer;
  /// frame_buffer.to_node(conduit_frame_buffer);
  /// conduit::relay::io::blueprint::save_mesh(conduit_frame_buffer, output_file + ".blueprint_root_hdf5");
}


TEST (dray_hilbert, dray_level_one)
{
  using namespace dray;
  SFC<2> sfc;
  for (int32 i0 = 0; i0 < 4; ++i0)
  {
    const int32 child_num0 = sfc.child_num(i0);
    fprintf(stdout, "  %d_ |  ", child_num0);
    for (int32 i1 = 0; i1 < 4; ++i1)
    {
      const int32 child_num1 = sfc.subcurve(i0).child_num(i1);
      fprintf(stdout, " %d%d", child_num0, child_num1);
    }
    fprintf(stdout, "\n");
  }
}


TEST (dray_hilbert, dray_table)
{
  using namespace dray;
  constexpr int32 dim = 3;
  constexpr int32 split_size = 1u << dim;

  SFC<dim> sfc;
  sfc::print<dim>(std::cout, sfc.orientation()) << "\n";
  sfc::print<dim>(std::cout, sfc::compose<dim>(sfc.orientation(), sfc.orientation())) << "\n";
  std::cout << "--------------------\n";

  std::cout << "      \tloc\t";
  std::cout << "perm\tinv\trefl" << "\n";

  for (int32 i0 = 0; i0 < split_size; ++i0)
  {
    const int32 child_num = sfc.child_num(i0);
    SFC<dim> sub_sfc = sfc.subcurve(i0);
    std::cout << i0 << "(" << std::bitset<dim>(i0) << ")" << "\t"
              << std::bitset<dim>(child_num) << "\t";
    sfc::print<dim>(std::cout, sub_sfc.orientation()) << "\n";
  }
}



template <int dim, typename EmitFunc>
void traverse_hilbert(const dray::TreeNode<dim> &subtree_root,
                      const dray::SFC<dim> &subtree_curve,
                      int levels,
                      EmitFunc emit)
{
  if (levels == 0)
    emit(subtree_root);
  else
  {
    for (int subindex = 0; subindex < (1u<<dim); ++subindex)
      traverse_hilbert(subtree_root.child(subtree_curve.child_num(subindex)),
                       subtree_curve.subcurve(subindex),
                       levels-1,
                       emit);
  }
}


template <int dim>
bool check_adjacent(int levels)
{
  using namespace dray;

  const Float len = 1.0f / (1u << levels);

  Vec<Float, dim> prev;
  bool inited = false;
  bool adjacent = true;

  traverse_hilbert(TreeNode<dim>::root(), SFC<dim>(), levels,
      [&]
      (const TreeNode<dim> &tree_node)
  {
    Vec<Float, dim> v = tree_node.coord().template vec<Float>();

    bool this_adjacent = true;
    if (inited)
    {
      Vec<Float, dim> diff = v - prev;
      int count_diff = 0;
      for (int d = 0; d < dim; ++d)
      {
        this_adjacent &= (diff[d] == 0 || abs(diff[d]) == len);
        count_diff += (diff[d] != 0);
      }
      this_adjacent &= (count_diff == 1);
    }
    adjacent &= this_adjacent;
    prev = v;
    inited = true;
  });

  return adjacent;
}


TEST (dray_hilbert, dray_hilbert_adjacent)
{
  constexpr int dim2 = 2;
  constexpr int dim3 = 3;
  constexpr int dim4 = 4;
  const int levels = 7;

  /// EXPECT_TRUE(check_adjacent<dim2>(levels));
  /// EXPECT_TRUE(check_adjacent<dim3>(levels));
  EXPECT_TRUE(check_adjacent<dim4>(levels));

  // About 25ns per leaf on my laptop.
}





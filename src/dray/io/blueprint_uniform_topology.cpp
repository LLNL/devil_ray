// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/io/blueprint_uniform_topology.hpp>

#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{
namespace detail
{

std::shared_ptr<UniformTopology>
import_topology_into_uniform(const conduit::Node &topo,
                             const conduit::Node &coords)
{
  using conduit::Node;

  const Node &n_dims = coords["dims"];

  // cell dims
  int dims_i = n_dims["i"].to_int() - 1;
  int dims_j = n_dims["j"].to_int() - 1;
  int dims_k = 1;
  bool is_2d = true;

  // check for 3d
  if(n_dims.has_path("k"))
  {
    dims_k = n_dims["k"].to_int() - 1;
    is_2d = false;
  }

  Float origin_x = 0.0f;
  Float origin_y = 0.0f;
  Float origin_z = 0.0f;

  if(coords.has_path("origin"))
  {
    const Node &n_origin = coords["origin"];

    if(n_origin.has_child("x"))
    {
      origin_x = n_origin["x"].to_float32();
    }

    if(n_origin.has_child("y"))
    {
      origin_y = n_origin["y"].to_float32();
    }

    if(n_origin.has_child("z"))
    {
      origin_z = n_origin["z"].to_float32();
    }
  }

  Float spacing_x = 1.0f;
  Float spacing_y = 1.0f;
  Float spacing_z = 1.0f;

  if(coords.has_path("spacing"))
  {
    const Node &n_spacing = coords["spacing"];

    if(n_spacing.has_path("dx"))
    {
        spacing_x = n_spacing["dx"].to_float32();
    }

    if(n_spacing.has_path("dy"))
    {
        spacing_y = n_spacing["dy"].to_float32();
    }

    if(n_spacing.has_path("dz"))
    {
        spacing_z = n_spacing["dz"].to_float32();
    }
  }

  Vec<Float,3> spacing{spacing_x, spacing_y, spacing_z};
  Vec<Float,3> origin{origin_x, origin_y, origin_z};
  Vec<int32,3> dims{dims_i, dims_j, dims_k};

  std::shared_ptr<UniformTopology> utopo
    = std::make_shared<UniformTopology>(spacing, origin, dims);

  return utopo;
}


} // namespace detail
} // namespace dray


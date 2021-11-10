// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/transport/uniform_partials.hpp>
#include <dray/transport/dda.hpp>

#include <dray/policies.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/device_array.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{
  static AABB<3> uniform_domain_aabb(const UniformTopology *mesh);

  static DRAY_EXEC Vec<Float, 3> clamp_ray_origin(
      const AABB<3> &aabb,
      const Vec<Float, 3> &ray_origin,
      const Vec<Float, 3> &ray_dir);

  std::pair<Array<Float>,
            Array<Vec<Float, 3>>>
  uniform_partials(
      const UniformTopology *mesh,
      const LowOrderField *absorption,
      Vec<Float, 3> &source,
      Array<Vec<Float, 3>> &world_points)
  {
    const int32 num_points = world_points.size();
    const int32 num_comp = world_points.ncomp();

    Array<Float> partial_opt_depth;
    Array<Vec<Float, 3>> partial_entry;
    partial_opt_depth.resize(num_points, num_comp);
    partial_entry.resize(num_points);

    ConstDeviceArray<Float> d_absorption = absorption->d_values();
    ConstDeviceArray<Vec<Float, 3>> d_world_points(world_points);
    NonConstDeviceArray<Float> d_partial_opt_depth(partial_opt_depth);
    NonConstDeviceArray<Vec<Float, 3>> d_partial_entry(partial_entry);
    const FS_DDATraversal dda(*mesh);
    AABB<3> domain_aabb = uniform_domain_aabb(mesh);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_points),
        [=] DRAY_LAMBDA (int32 point_idx)
    {
      Vec<Float, 3> begin = source;
      Vec<Float, 3> end = d_world_points.get_item(point_idx);
      Vec<Float, 3> dir = (end - begin).normalized();
      begin = clamp_ray_origin(domain_aabb, begin, dir);
      Float distance_left = (end - begin).magnitude();

      FS_TraversalState state;
      dda.init_traversal(begin, dir, state);

      Float distance = 0.f;
      Float *res = new Float[num_comp];
      for (int32 comp = 0; comp < num_comp; ++comp)
        res[comp] = 0.f;

      while (dda.is_inside(state.m_voxel) && distance_left > 0.0f)
      {
        const Float voxel_exit = state.exit();
        Float length = voxel_exit - distance;
        length = min(length, distance_left);

        const int32 cell_id = dda.voxel_index(state.m_voxel);
        for (int32 component = 0; component < num_comp; ++component)
        {
          const Float segment_optical_depth
              = length * d_absorption.get_item(cell_id, component);

          res[component] += segment_optical_depth;
        }

        distance_left -= length;
        distance = voxel_exit;
        state.advance();
      }

      for (int32 component = 0; component < num_comp; ++component)
        d_partial_opt_depth.get_item(point_idx, component) = res[component];
      d_partial_entry.get_item(point_idx) = begin;

      delete [] res;
    });

    return std::make_pair(partial_opt_depth, partial_entry);
  }

  // uniform_domain_aabb()
  static AABB<3> uniform_domain_aabb(const UniformTopology *mesh)
  {
    const Vec<int32, 3> cell_dims = mesh->cell_dims();
    const Vec<Float, 3> origin = mesh->origin();

    // [Diagonal] = [Spacing] .(entry-wise product) [Cell Dims]
    Vec<Float, 3> diagonal = mesh->spacing();
    for (int32 d = 0; d < 3; ++d)
      diagonal[d] *= cell_dims[d];

    // AABB covers lower corner and upper corner.
    AABB<3> aabb;
    aabb.include(origin);
    aabb.include(origin + diagonal);

    return aabb;
  }


  // clamp_ray_origin()
  static DRAY_EXEC Vec<Float, 3> clamp_ray_origin(
      const AABB<3> &aabb,
      const Vec<Float, 3> &ray_origin,
      const Vec<Float, 3> &ray_dir)
  {
    if (aabb.contains(ray_origin))
      return ray_origin;

    Float intercept = 0;  // Only go in front of the ray origin.
    Vec<Float, 3> v0 = aabb.min() - ray_origin;
    Vec<Float, 3> v1 = aabb.max() - ray_origin;
    for (int32 d = 0; d < 3; ++d)
    {
      v0[d] *= rcp_safe(ray_dir[d]);
      v1[d] *= rcp_safe(ray_dir[d]);
      Float near_plane = min(v0[d], v1[d]);
      if (intercept < near_plane)
        intercept = near_plane;  // max of "mins"
    }

    return ray_origin + ray_dir * intercept;
  }



}

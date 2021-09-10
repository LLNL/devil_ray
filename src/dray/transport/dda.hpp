// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DDA_HPP
#define DRAY_DDA_HPP

#include <dray/uniform_topology.hpp>

namespace dray
{
  struct FS_TraversalState
  {
    Vec<Float,3> m_delta_max;
    Vec<Float,3> m_delta;
    Vec<int32,3> m_voxel;
    Vec<Float,3> m_dir;

    // distance to voxel exit from initial point
    DRAY_EXEC
    Float exit() const
    {
      return min(m_delta_max[0], min(m_delta_max[1], m_delta_max[2]));
    }

    // advances to the next voxel along the ray
    DRAY_EXEC void advance()
    {
      int32 advance_dir = 0;
      for(int32 i = 1; i < 3; ++i)
      {
        if(m_delta_max[i] < m_delta_max[advance_dir])
        {
          advance_dir = i;
        }
      }
      m_delta_max[advance_dir] += m_delta[advance_dir];
      m_voxel[advance_dir] += m_dir[advance_dir] < 0.f ? -1 : 1;
    }
  };

  struct FS_DDATraversal
  {
    const Vec<int32,3> m_dims;
    const Vec<Float,3> m_origin;
    const Vec<Float,3> m_spacing;

    FS_DDATraversal(UniformTopology &topo)
      : m_dims(topo.cell_dims()),
        m_origin(topo.origin()),
        m_spacing(topo.spacing())
    {
    }

    DRAY_EXEC
    bool is_inside(const Vec<int32, 3>& index) const
    {
      bool inside = true;
      const int32 minIndex = min(index[0], min(index[1], index[2]));
      if(minIndex < 0) inside = false;
      if(index[0] >= m_dims[0]) inside = false;
      if(index[1] >= m_dims[1]) inside = false;
      if(index[2] >= m_dims[2]) inside = false;
      return inside;
    }

    DRAY_EXEC
    int32 voxel_index(const Vec<int32, 3> &voxel) const
    {
      return voxel[0] + voxel[1] * m_dims[0] + voxel[2] * m_dims[0] * m_dims[1];
    }

    DRAY_EXEC Float
    init_traversal(const Vec<Float,3> &point,
                   const Vec<Float,3> &dir,
                   FS_TraversalState &state) const
    {
      Vec<Float, 3> temp = point;
      temp = temp - m_origin;
      state.m_voxel[0] = temp[0] / m_spacing[0];
      state.m_voxel[1] = temp[1] / m_spacing[1];
      state.m_voxel[2] = temp[2] / m_spacing[2];
      state.m_dir = dir;

      Vec<Float,3> step;
      step[0] = (dir[0] >= 0.f) ? 1.f : -1.f;
      step[1] = (dir[1] >= 0.f) ? 1.f : -1.f;
      step[2] = (dir[2] >= 0.f) ? 1.f : -1.f;

      Vec<Float,3> next_boundary;
      next_boundary[0] = (Float(state.m_voxel[0]) + step[0]) * m_spacing[0];
      next_boundary[1] = (Float(state.m_voxel[1]) + step[1]) * m_spacing[1];
      next_boundary[2] = (Float(state.m_voxel[2]) + step[2]) * m_spacing[2];

      // correct next boundary for negative directions
      if(step[0] == -1.f) next_boundary[0] += m_spacing[0];
      if(step[1] == -1.f) next_boundary[1] += m_spacing[1];
      if(step[2] == -1.f) next_boundary[2] += m_spacing[2];

      // distance to next voxel boundary
      state.m_delta_max[0] = (dir[0] != 0.f) ?
        (next_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

      state.m_delta_max[1] = (dir[1] != 0.f) ?
        (next_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

      state.m_delta_max[2] = (dir[2] != 0.f) ?
        (next_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

      // distance along ray to traverse x,y, and z of a voxel
      state.m_delta[0] = (dir[0] != 0) ? m_spacing[0] / dir[0] * step[0] : infinity<Float>();
      state.m_delta[1] = (dir[1] != 0) ? m_spacing[1] / dir[1] * step[1] : infinity<Float>();
      state.m_delta[2] = (dir[2] != 0) ? m_spacing[2] / dir[2] * step[2] : infinity<Float>();

      Vec<Float,3> exit_boundary;
      exit_boundary[0] = step[0] < 0.f ? 0.f : Float(m_dims[0]) * m_spacing[0];
      exit_boundary[1] = step[1] < 0.f ? 0.f : Float(m_dims[1]) * m_spacing[1];
      exit_boundary[2] = step[2] < 0.f ? 0.f : Float(m_dims[2]) * m_spacing[2];

      // Masado questions these lines
      if(step[0] == -1.f) exit_boundary[0] += m_spacing[0];
      if(step[1] == -1.f) exit_boundary[1] += m_spacing[1];
      if(step[2] == -1.f) exit_boundary[2] += m_spacing[2];

      Vec<Float,3> exit_dist;
      // distance to grid exit
      exit_dist[0] = (dir[0] != 0.f) ?
        (exit_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

      exit_dist[1] = (dir[1] != 0.f) ?
        (exit_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

      exit_dist[2] = (dir[2] != 0.f) ?
        (exit_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

      //std::cout<<"Init voxel "<<voxel<<"\n";

      return min(exit_dist[0], min(exit_dist[1], exit_dist[2]));
    }
  };

}

#endif//DRAY_DDA_HPP

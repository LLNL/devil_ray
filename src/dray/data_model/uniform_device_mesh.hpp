// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_DEVICE_MESH_HPP
#define DRAY_UNIFORM_DEVICE_MESH_HPP

#include <dray/data_model/uniform_mesh.hpp>
#include <dray/exports.hpp>
#include <dray/location.hpp>
#include <dray/vec.hpp>
#include <dray/ray.hpp>

namespace dray
{

/*
 * @class DeviceMesh
 * @brief Device-safe access to a collection of elements
 * (just knows about the geometry, not fields).
 */
struct UniformDeviceMesh
{
  Vec<Float,3> m_origin;
  Vec<Float,3> m_spacing;
  Vec<int32,3> m_cell_dims;
  Vec<Float,3> m_max_point;

  UniformDeviceMesh(UniformMesh &mesh);
  UniformDeviceMesh () = delete;

  //DRAY_EXEC ElemT get_elem (int32 el_idx) const;
  DRAY_EXEC Location locate (const Vec<Float, 3> &point) const;
  DRAY_EXEC Float hit(const Ray &ray) const;

  DRAY_EXEC Vec<int32,3> logical_cell_index(const int32 &cell_id) const;
  DRAY_EXEC Vec<Float,3> cell_center(const int32 &cell_id) const;
  DRAY_EXEC bool is_inside(const Vec<Float,3> &point) const;
};

UniformDeviceMesh::UniformDeviceMesh(UniformMesh &mesh)
  : m_origin(mesh.origin()),
    m_spacing(mesh.spacing()),
    m_cell_dims(mesh.cell_dims())
{
  Vec<Float,3> unit_length;
  unit_length[0] = static_cast<Float>(m_cell_dims[0]);
  unit_length[1] = static_cast<Float>(m_cell_dims[1]);
  unit_length[2] = static_cast<Float>(m_cell_dims[2]);
  m_max_point[0] = m_origin[0] + m_spacing[0] * unit_length[0];
  m_max_point[1] = m_origin[1] + m_spacing[1] * unit_length[1];
  m_max_point[2] = m_origin[2] + m_spacing[2] * unit_length[2];
}

DRAY_EXEC Float UniformDeviceMesh::hit(const Ray &ray) const
{
  Float dirx = ray.m_dir[0];
  Float diry = ray.m_dir[1];
  Float dirz = ray.m_dir[2];
  Float origx = ray.m_orig[0];
  Float origy = ray.m_orig[1];
  Float origz = ray.m_orig[2];

  const Float inv_dirx = rcp_safe (dirx);
  const Float inv_diry = rcp_safe (diry);
  const Float inv_dirz = rcp_safe (dirz);

  const Float odirx = origx * inv_dirx;
  const Float odiry = origy * inv_diry;
  const Float odirz = origz * inv_dirz;

  const Float xmin = m_origin[0] * inv_dirx - odirx;
  const Float ymin = m_origin[1] * inv_diry - odiry;
  const Float zmin = m_origin[2] * inv_dirz - odirz;
  const Float xmax = m_max_point[0] * inv_dirx - odirx;
  const Float ymax = m_max_point[1] * inv_diry - odiry;
  const Float zmax = m_max_point[2] * inv_dirz - odirz;

  const Float min_int = ray.m_near;
  Float min_dist =
    max (max (max (min (ymin, ymax), min (xmin, xmax)), min (zmin, zmax)), min_int);
  Float max_dist = min (min (max (ymin, ymax), max (xmin, xmax)), max (zmin, zmax));
  max_dist = min(max_dist, ray.m_far);

  Float dist = infinity<Float>();
  if (max_dist > min_dist)
  {
    dist = min_dist;
  }

  return dist;
}

DRAY_EXEC Vec<int32,3>
UniformDeviceMesh::logical_cell_index(const int32 &cell_id) const
{
  Vec<int32,3> idx;
  idx[0] = cell_id % (m_cell_dims[0]);
  idx[1] = (cell_id/ (m_cell_dims[0])) % (m_cell_dims[1]);
  idx[2] = cell_id / ((m_cell_dims[0]) * (m_cell_dims[1]));
  return idx;
}

DRAY_EXEC Vec<Float,3>
UniformDeviceMesh::cell_center(const int32 &cell_id) const
{
  Vec<int32,3> idx = logical_cell_index(cell_id);
  Vec<Float,3> center;
  center[0] = m_origin[0] + Float(idx[0]) * m_spacing[0] + m_spacing[0] * 0.5f;
  center[1] = m_origin[1] + Float(idx[1]) * m_spacing[1] + m_spacing[1] * 0.5f;
  center[2] = m_origin[2] + Float(idx[2]) * m_spacing[2] + m_spacing[2] * 0.5f;
  return center;
}

DRAY_EXEC bool
UniformDeviceMesh::is_inside(const Vec<Float,3> &point) const
{
  bool inside = true;
  if (point[0] < m_origin[0] || point[0] > m_max_point[0])
  {
    inside = false;
  }
  if (point[1] < m_origin[1] || point[1] > m_max_point[1])
  {
    inside = false;
  }
  if (point[2] < m_origin[2] || point[2] > m_max_point[2])
  {
    inside = false;
  }
  return inside;
}

DRAY_EXEC Location
UniformDeviceMesh::locate(const Vec<Float,3> &point) const
{
  Location loc;
  loc.m_cell_id = -1;
  loc.m_ref_pt = {{0.f, 0.f, 0.f}};
  if(is_inside(point))
  {
    Vec<Float,3> temp = point;
    temp = temp - m_origin;
    temp[0] /= m_spacing[0];
    temp[1] /= m_spacing[1];
    temp[2] /= m_spacing[2];
    //make sure that if we border the upper edge we return
    // a consistent cell
    if (temp[0] == Float(m_cell_dims[0]))
      temp[0] = Float(m_cell_dims[0] - 1);
    if (temp[1] == Float(m_cell_dims[1]))
      temp[1] = Float(m_cell_dims[1] - 1);
    if (temp[2] == Float(m_cell_dims[2]))
      temp[2] = Float(m_cell_dims[2] - 1);
    Vec<int32,3> cell;
    for(int32 i = 0; i < 3; ++i)
    {
      Float int_part;
      // fractional part will be the parametric coordinate
      loc.m_ref_pt[i] = modf(temp[i], &int_part);
      // int part is the cell
      cell[i] = (int32) int_part;
    }

    loc.m_cell_id = (cell[2] * m_cell_dims[1] + cell[1]) * m_cell_dims[0] + cell[0];

  }
  return loc;
}

} // namespace dray


#endif

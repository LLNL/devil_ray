// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/uniform_topology.hpp>
#include <dray/error.hpp>

namespace dray
{
UniformTopology::UniformTopology(const Vec<Float,3> &spacing,
                                 const Vec<Float,3> &origin,
                                 const Vec<int32,3> &dims)
  : m_spacing(spacing),
    m_origin(origin),
    m_dims(dims)
{
}

UniformTopology::~UniformTopology()
{

}

int32
UniformTopology::cells() const
{
  return m_dims[0] * m_dims[1] * m_dims[2];
}

int32
UniformTopology::order() const
{
  return 1;
}

int32
UniformTopology::dims() const
{
  return 3;
}

std::string
UniformTopology::type_name() const
{
  return "uniform";
}

AABB<3>
UniformTopology::bounds()
{
  AABB<3> bounds;
  bounds.include(m_origin);
  Vec<Float,3> upper;
  upper[0] = m_origin[0] + m_spacing[0] * Float(m_dims[0]);
  upper[1] = m_origin[1] + m_spacing[1] * Float(m_dims[1]);
  upper[2] = m_origin[2] + m_spacing[2] * Float(m_dims[2]);
  bounds.include(upper);
  return bounds;
}

Array<Location>
UniformTopology::locate(Array<Vec<Float, 3>> &wpoints)
{
  DRAY_ERROR("not implemented");
}


void UniformTopology::to_node(conduit::Node &n_topo)
{
  n_topo.reset();
  n_topo["type_name"] = type_name();
  /// n_topo["order"] = m_mesh.get_poly_order();

  throw std::logic_error(("Not implemented to_node()! " __FILE__));

  /// conduit::Node &n_gf = n_topo["grid_function"];
  /// GridFunction<3u> gf = m_mesh.get_dof_data();
  /// gf.to_node(n_gf);
}


Vec<int32,3>
UniformTopology::cell_dims() const
{
  return m_dims;
}

Vec<Float,3>
UniformTopology::spacing() const
{
  return m_spacing;
}

Vec<Float,3>
UniformTopology::origin() const
{
  return m_origin;
}

void UniformTopology::to_blueprint(conduit::Node &n_dataset)
{
  // hard coded topology and coords names;
  const std::string topo_name = "topo";
  const std::string coord_name = "coords";

  conduit::Node &n_topo = n_dataset["topologies/"+topo_name];
  n_topo["coordset"] = coord_name;
  n_topo["type"] = "uniform";

  conduit::Node &n_coords = n_dataset["coordsets/"+coord_name];
  n_coords["type"] = "uniform";
  n_coords["dims/i"] = m_dims[0] + 1;
  n_coords["dims/j"] = m_dims[1] + 1;
  n_coords["dims/k"] = m_dims[2] + 1;

  n_coords["origin/x"] = m_origin[0];
  n_coords["origin/y"] = m_origin[1];
  n_coords["origin/z"] = m_origin[2];

  n_coords["spacing/dx"] = m_spacing[0];
  n_coords["spacing/dy"] = m_spacing[1];
  n_coords["spacing/dz"] = m_spacing[2];
}

} // namespace dray

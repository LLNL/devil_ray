// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/uniform_topology.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/aabb.hpp>
#include <RAJA/RAJA.hpp>

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
  DRAY_LOG_OPEN ("locate");

  const int32 size = wpoints.size ();
  Array<Location> locations;
  locations.resize (size);

  Location *loc_ptr = locations.get_device_ptr ();
  const Vec<Float,3> *points_ptr = wpoints.get_device_ptr_const();

  Vec<int32, 3> dims = this->cell_dims();
  Vec<Float, 3> spacing = this->spacing();
  Vec<Float, 3> origin = this->origin();

  AABB<3> mesh_bounds = this->bounds();
  AABB<3> mesh_bounds_eps = mesh_bounds;
  mesh_bounds_eps.expand(epsilon<Float>());

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 index)
  {
    Location loc = { -1, {{ -1.f, -1.f, -1.f }} };  // default if fail to locate

    Vec<Float, 3> point = points_ptr[index];
    if (mesh_bounds_eps.contains(point))
    {
      mesh_bounds.clamp(point);
      Vec<Float, 3> relative = point - origin;  // save it for ref coords
      loc = detail::uniform_locate_float(relative, dims, spacing);
    }

    loc_ptr[index] = loc;
  });


  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();

  return locations;
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


// evaluator()
UniformTopology::Evaluator
UniformTopology::evaluator() const
{
  return Evaluator{m_spacing, m_origin, m_dims};
}

// jacobian_evaluator()
UniformTopology::JacobianEvaluator
UniformTopology::jacobian_evaluator() const
{
  Vec<Vec<Float, 3>, 3> uniform_jacobian =
  {{ {{ m_spacing[0], 0, 0 }},
     {{ 0, m_spacing[1], 0}},
     {{ 0, 0, m_spacing[2]}} }};

  return JacobianEvaluator{ uniform_jacobian };
}

void UniformTopology::to_blueprint(conduit::Node &n_dataset)
{
  // hard coded topology and coords names;
  const std::string topo_name = this->name();
  const std::string coord_name = "coords_"+topo_name;

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

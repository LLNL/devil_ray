// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/point_topology.hpp>
#include <dray/error.hpp>

#include <dray/error_check.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/policies.hpp>
#include <dray/linear_bvh_builder.hpp>

namespace dray
{

namespace detail
{

Array<AABB<3>> extract_sphere_aabbs(Array<Vec<Float,3>> points, Array<Float> radii)
{
  const int32 size = points.size();
  Array<AABB<3>> aabbs;
  aabbs.resize(size);

  AABB<3> *aabb_ptr = aabbs.get_device_ptr();
  const Vec<Float,3> *point_ptr = points.get_device_ptr_const();
  const Float *radii_ptr = radii.get_device_ptr_const();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    const Vec<Float,3> point = point_ptr[ii];
    const Float radius = radii_ptr[ii];
    AABB<3> bounds;

    Vec<Float,3> temp;
    temp[0] = radius;
    temp[1] = 0;
    temp[2] = 0;

    bounds.include(point + temp);
    bounds.include(point - temp);

    temp[0] = 0;
    temp[1] = radius;
    temp[2] = 0;

    bounds.include(point + temp);
    bounds.include(point - temp);

    temp[0] = 0;
    temp[1] = 0;
    temp[2] = radius;

    bounds.include(point + temp);
    bounds.include(point - temp);
    aabb_ptr[ii] = bounds;
  });
  return aabbs;
}

} // namespace detail


PointTopology::PointTopology(Array<Vec<Float,3>> points, Array<Float> radii)
  : m_points(points),
    m_radii(radii)
{
  if(m_points.size() != m_radii.size())
  {
    DRAY_ERROR("Points and radii must have the same size");
  }

  const int32 size = m_points.size();
  Array<AABB<3>> aabbs = detail::extract_sphere_aabbs(m_points, m_radii);
  LinearBVHBuilder builder;
  m_bvh = builder.construct(aabbs);
}

PointTopology::~PointTopology()
{

}

int32
PointTopology::cells() const
{
  return m_points.size();
}

int32
PointTopology::order() const
{
  return 0;
}

int32
PointTopology::dims() const
{
  return 1;
}

std::string
PointTopology::type_name() const
{
  return "point_topology";
}

AABB<3>
PointTopology::bounds()
{
  return m_bvh.m_bounds;
}

Array<Location>
PointTopology::locate(Array<Vec<Float, 3>> &wpoints)
{
  DRAY_ERROR("locate not implemented");
  Array<Location> loc;
  return loc;
}

void
PointTopology::to_node(conduit::Node &n_topo)
{
  DRAY_ERROR("to_node not implemented");
}

Array<Vec<Float,3>>
PointTopology::points()
{
  return m_points;
}

Array<Float>
PointTopology::radii()
{
  return m_radii;
}

BVH
PointTopology::bvh()
{
  return m_bvh;
}


} // namespace dray

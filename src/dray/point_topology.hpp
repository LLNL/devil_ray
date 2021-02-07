// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POINT_TOPOLOGY_HPP
#define DRAY_POINT_TOPOLOGY_HPP

#include <dray/topology_base.hpp>
#include <dray/vec.hpp>
#include <dray/bvh.hpp>

namespace dray
{

class PointTopology : public TopologyBase
{
protected:
  Array<Vec<Float,3>> m_points;
  Array<Float> m_radii;
  BVH m_bvh;
public:
  PointTopology() = delete;
  PointTopology(Array<Vec<Float,3>> points, Array<Float> radii);

  virtual ~PointTopology();
  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  virtual std::string type_name() const override;

  virtual AABB<3> bounds() override;
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) override;
  virtual void to_node(conduit::Node &n_topo) override;


  Array<Vec<Float,3>> points();
  Array<Float> radii();
  BVH bvh();

};
} // namespace dray

#endif // DRAY_POINT_TOPOLOGY

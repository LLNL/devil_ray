// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_TOPOLOGY_HPP
#define DRAY_UNIFORM_TOPOLOGY_HPP

#include <dray/topology_base.hpp>

namespace dray
{

class UniformTopology : public TopologyBase
{

protected:
  Vec<Float, 3> m_spacing;
  Vec<Float, 3> m_origin;
  Vec<int32, 3> m_dims;
public:
  UniformTopology() = delete;
  UniformTopology(const Vec<Float,3> &spacing,
                  const Vec<Float,3> &m_origin,
                  const Vec<int32,3> &dims);

  virtual ~UniformTopology();
  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  Vec<int32,3> cell_dims() const;
  Vec<Float,3> spacing() const;
  Vec<Float,3> origin() const;

  virtual std::string type_name() const override;

  // bounds() should be const, but DerivedTopology/Mesh needs mutable.
  virtual AABB<3> bounds() override;

  // locate() should be const, but DerivedTopology/Mesh needs mutable.
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) override;

  virtual void to_node(conduit::Node &n_topo) override;

  virtual void to_blueprint(conduit::Node &n_dataset) override;
};

} // namespace dray

#endif // DRAY_REF_POINT_HPP

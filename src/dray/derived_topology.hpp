// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DERIVED_TOPOLOGY_HPP
#define DRAY_DERIVED_TOPOLOGY_HPP

#include <dray/topology_base.hpp>

namespace dray
{

template<typename Element>
class DerivedTopology : public TopologyBase
{
protected:
  Mesh<Element> m_mesh;
public:
  DerivedTopology() = delete;
  DerivedTopology(Mesh<Element> &mesh);
  virtual ~DerivedTopology();
  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  virtual std::string shape_name() const override;

  virtual AABB<3> bounds() const override;
  Mesh<Element>& mesh();
};

// Element<topo dims, ncomps, base_shape, polynomial order>
using Hex3  = Element<3u, 3u, ElemType::Quad, Order::General>;
using Quad3 = Element<2u, 3u,ElemType::Quad, Order::General>;

using HexTopology = DerivedTopology<Hex3>;
using QuadTopology = DerivedTopology<Quad3>;

} // namespace dray

#endif // DRAY_REF_POINT_HPP

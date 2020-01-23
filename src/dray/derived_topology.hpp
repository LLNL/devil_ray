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

  using ElementType = Element;

  virtual ~DerivedTopology();
  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  virtual std::string type_name() const override;

  virtual AABB<3> bounds() const override;
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) const override;
  virtual Array<Vec<Float, 3>> eval_location (Array<Location> &rpoints) const override;
  Mesh<Element>& mesh();
};

// Element<topo dims, ncomps, base_shape, polynomial order>
using Hex3  = Element<3u, 3u, ElemType::Quad, Order::General>;
using HexP1  = Element<3u, 3u, ElemType::Quad, Order::Linear>;
using Quad3 = Element<2u, 3u,ElemType::Quad, Order::General>;

using HexTopology = DerivedTopology<Hex3>;
using QuadTopology = DerivedTopology<Quad3>;


// Design Consideration: same orders
//template <class order>
//struct HexTopology
//{
//  typedef Element<3u, 3u, ElemType::Quad, order> ElemType;
//  typedef ScalarElement<3u, 1u, ElemType::Quad, order> ElemType;
//  typedef VectorElement<3u, 3u, ElemType::Quad, order> ElemType;
//}

} // namespace dray

#endif // DRAY_REF_POINT_HPP

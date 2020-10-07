// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DERIVED_TOPOLOGY_HPP
#define DRAY_DERIVED_TOPOLOGY_HPP

#include <dray/topology_base.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/mesh.hpp>

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
  DerivedTopology(Mesh<Element> &&mesh);

  using ElementType = Element;

  virtual ~DerivedTopology();
  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  virtual std::string type_name() const override;

  virtual AABB<3> bounds() const override;
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) const override;
  virtual void to_node(conduit::Node &n_topo) override;

  Mesh<Element>& mesh();
  const Mesh<Element>& mesh() const;
};

// Element<topo dims, ncomps, base_shape, polynomial order>
using Hex3   = Element<3u, 3u, ElemType::Tensor, Order::General>;
using Hex_P1 = Element<3u, 3u, ElemType::Tensor, Order::Linear>;
using Hex_P2 = Element<3u, 3u, ElemType::Tensor, Order::Quadratic>;

using Tet3   = Element<3u, 3u, ElemType::Simplex, Order::General>;
using Tet_P1 = Element<3u, 3u, ElemType::Simplex, Order::Linear>;
using Tet_P2 = Element<3u, 3u, ElemType::Simplex, Order::Quadratic>;

using Quad3   = Element<2u, 3u,ElemType::Tensor, Order::General>;
using Quad_P1 = Element<2u, 3u,ElemType::Tensor, Order::Linear>;
using Quad_P2 = Element<2u, 3u,ElemType::Tensor, Order::Quadratic>;

using Tri3    = Element<2u, 3u, ElemType::Simplex, Order::General>;
using Tri_P1  = Element<2u, 3u, ElemType::Simplex, Order::Linear>;
using Tri_P2  = Element<2u, 3u, ElemType::Simplex, Order::Quadratic>;

using HexTopology = DerivedTopology<Hex3>;
using HexTopology_P1 = DerivedTopology<Hex_P1>;
using HexTopology_P2 = DerivedTopology<Hex_P2>;

using TetTopology = DerivedTopology<Tet3>;
using TetTopology_P1 = DerivedTopology<Tet_P1>;
using TetTopology_P2 = DerivedTopology<Tet_P2>;

using QuadTopology = DerivedTopology<Quad3>;
using QuadTopology_P1 = DerivedTopology<Quad_P1>;
using QuadTopology_P2 = DerivedTopology<Quad_P2>;

using TriTopology = DerivedTopology<Tri3>;
using TriTopology_P1 = DerivedTopology<Tri_P1>;
using TriTopology_P2 = DerivedTopology<Tri_P2>;


// Design Consideration: same orders
//template <class order>
//struct HexTopology
//{
//  typedef Element<3u, 3u, ElemType::Tensor, order> ElemType;
//  typedef ScalarElement<3u, 1u, ElemType::Tensor, order> ElemType;
//  typedef VectorElement<3u, 3u, ElemType::Tensor, order> ElemType;
//}

} // namespace dray

#endif // DRAY_REF_POINT_HPP

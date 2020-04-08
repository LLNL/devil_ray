// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/derived_topology.hpp>

namespace dray
{

template<typename Element>
DerivedTopology<Element>::DerivedTopology(Mesh<Element> &&mesh)
  : m_mesh(mesh)
{
}

template<typename Element>
DerivedTopology<Element>::DerivedTopology(Mesh<Element> &mesh)
  : m_mesh(mesh)
{
}

template<typename Element>
DerivedTopology<Element>::~DerivedTopology()
{
};

template<typename Element>
int32 DerivedTopology<Element>::cells() const
{
  return m_mesh.get_num_elem();
}

template<typename Element>
int32 DerivedTopology<Element>::order() const
{
  return m_mesh.get_poly_order();
}

template<typename Element>
AABB<3> DerivedTopology<Element>::bounds() const
{
  return m_mesh.get_bounds();
}

template<typename Element>
int32 DerivedTopology<Element>::dims() const
{
  return Mesh<Element>::dim;
}

template<typename Element>
std::string DerivedTopology<Element>::type_name() const
{
  return element_name<Element>(Element());
}

template<typename Element>
Array<Location>
DerivedTopology<Element>::locate(Array<Vec<Float, 3>> &wpoints) const
{
  return m_mesh.locate(wpoints);
}

template<typename Element>
Mesh<Element>& DerivedTopology<Element>::mesh()
{
  return m_mesh;
}

// Currently supported topologies
template class DerivedTopology<Hex3>;
template class DerivedTopology<Tet3>;
template class DerivedTopology<Quad3>;
template class DerivedTopology<Tri3>;
template class DerivedTopology<Hex_P1>;
template class DerivedTopology<Hex_P2>;
template class DerivedTopology<Quad_P1>;
template class DerivedTopology<Quad_P2>;


} // namespace dray

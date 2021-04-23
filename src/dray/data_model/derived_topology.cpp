// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_model/derived_topology.hpp>

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
AABB<3> DerivedTopology<Element>::bounds()
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
  return element_name<Element>();
}

template<typename Element>
Array<Location>
DerivedTopology<Element>::locate(Array<Vec<Float, 3>> &wpoints)
{
  return m_mesh.locate(wpoints);
}

template<typename Element>
Mesh<Element>& DerivedTopology<Element>::mesh()
{
  return m_mesh;
}

template<typename Element>
const Mesh<Element>& DerivedTopology<Element>::mesh() const
{
  return m_mesh;
}

template<typename Element>
void DerivedTopology<Element>::to_node(conduit::Node &n_topo)
{
  n_topo.reset();
  n_topo["type_name"] = type_name();
  n_topo["order"] = m_mesh.get_poly_order();

  conduit::Node &n_gf = n_topo["grid_function"];
  GridFunction<3u> gf = m_mesh.get_dof_data();
  gf.to_node(n_gf);

}

// Currently supported topologies
template class DerivedTopology<Hex3>;
template class DerivedTopology<Hex_P1>;
template class DerivedTopology<Hex_P2>;

template class DerivedTopology<Tet3>;
template class DerivedTopology<Tet_P1>;
template class DerivedTopology<Tet_P2>;

template class DerivedTopology<Quad3>;
template class DerivedTopology<Quad_P1>;
template class DerivedTopology<Quad_P2>;

template class DerivedTopology<Tri3>;
template class DerivedTopology<Tri_P1>;
template class DerivedTopology<Tri_P2>;



} // namespace dray

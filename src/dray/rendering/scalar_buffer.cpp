// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_buffer.hpp>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

namespace dray
{

conduit::Node * ScalarBuffer::to_node()
{
  conduit::Node *res = new conduit::Node();
  conduit::Node &mesh = *res;
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = m_width + 1;
  mesh["coordsets/coords/dims/j"] = m_height + 1;

  mesh["topologies/topo/coordset"] = "coords";
  mesh["topologies/topo/type"] = "uniform";

  for(int i = 0; i < m_names.size(); ++i)
  {
    const std::string path = "fields/" + m_names[i] + "/";
    mesh[path + "association"] = "element";
    mesh[path + "topology"] = "topo";
    const int size = m_scalars[i].size();
    const float32 *scalars = m_scalars[i].get_host_ptr_const();
    mesh[path + "values"].set(scalars, size);
  }

  mesh["fields/depth/association"] = "element";
  mesh["fields/depth/topology"] = "topo";
  const int size = m_depths.size();
  const float32 *depths = m_depths.get_host_ptr_const();
  mesh["fields/depth/values"].set(depths, size);

  conduit::Node verify_info;
  bool ok = conduit::blueprint::mesh::verify(mesh,verify_info);
  if(!ok)
  {
    verify_info.print();
  }
  return res;
}

} // namespace dray

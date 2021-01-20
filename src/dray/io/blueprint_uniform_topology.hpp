// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#ifndef DRAY_BLUEPRINT_UNIFORM_TOPOLOGY_HPP
#define DRAY_BLUEPRINT_UNIFORM_TOPOLOGY_HPP

#include <memory>
#include <conduit.hpp>
#include <dray/uniform_topology.hpp>

namespace dray
{
  namespace detail
  {
    std::shared_ptr<UniformTopology>
    import_topology_into_uniform(const conduit::Node &topo,
                                 const conduit::Node &coords);
  }
}


#endif//DRAY_BLUEPRINT_UNIFORM_TOPOLOGY_HPP

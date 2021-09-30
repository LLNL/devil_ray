// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TOPOLOGICAL_ORDER_HPP
#define DRAY_TOPOLOGICAL_ORDER_HPP

#include <dray/graph/cpu_graph.hpp>

namespace dray
{
  /**
   * topological_order()
   *
   *   Produces an ordering from a minimal node to a maximal node,
   *   where edges obtain a direction using less_than().
   *   The comparator
   *     [bool less_than(Node a, Node b) -> true if a < b]
   *   transforms an undirected graph to a directed graph.
   *   The resulting directed graph must be acyclic
   *   in order for a valid topological ordering to exist.
   *
   *   The original use case for imposing directionality
   *   _separately_ on top of an _undirected_ graph, is where
   *   the connectedness of domains is fixed, but their ordering
   *   depends on a point source location amongst the domains.
   *
   *   Note: For now the nodes must be int32's,
   *   but this could be easily changed with another template.
   */
  template <typename LessThan>
  std::vector<int32> topological_order(
      const portgraph::PortGraph<int32> &graph,  // should be undirected
      const LessThan & less_than);



  // =========================================
  // Implemetation
  // =========================================

  namespace detail
  {
    class TopologicalOrder
    {
      public:
        template <typename LessThan>
        static std::vector<int32> ordering(
            const portgraph::PortGraph<int32> &graph,
            const LessThan & less_than)
        {
          const int32 num_nodes = graph.num_nodes();
          TopologicalOrder builder(num_nodes);
          for (int32 node = 0; node < num_nodes; ++node)
            builder.up_to_and_including(node, graph, less_than);
          return builder.m_build_ordering;
        }

      private:
        std::vector<bool> m_pending;
        std::vector<bool> m_expired;
        std::vector<int32> m_build_ordering;

        TopologicalOrder(int32 num_nodes)
          : m_pending(num_nodes, false),
            m_expired(num_nodes, false),
            m_build_ordering(0)
        { }

        template <typename LessThan>
        void up_to_and_including(
            int32 goal_node,
            const portgraph::PortGraph<int32> &graph,
            const LessThan & less_than)
        {
          using namespace portgraph;
          if ((!m_pending[goal_node]) & (!m_expired[goal_node]))
          {
            m_pending[goal_node] = true;
            for (const Link<int32> & link : graph.from_node(goal_node))
              if (less_than(link.to.node, goal_node))
                up_to_and_including(link.to.node, graph, less_than);

            m_pending[goal_node] = false;
            m_expired[goal_node] = true;
            m_build_ordering.push_back(goal_node);
          }
        }
    };
  }


  // topological_order()
  template <typename LessThan>
  std::vector<int32> topological_order(const portgraph::PortGraph<int32> &graph,
                                       const LessThan & less_than)
  {
    return detail::TopologicalOrder::ordering(graph, less_than);
  }
}

#endif//DRAY_TOPOLOGICAL_ORDER_HPP

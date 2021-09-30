// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_CPU_GRAPH_HPP
#define DRAY_CPU_GRAPH_HPP

#include <map>
#include <set>

#include <dray/types.hpp>

namespace dray
{
  namespace portgraph
  {
    // From, To, NodePort
    template <typename KeyT>  struct From { KeyT node; int32 port; };
    template <typename KeyT>  struct To { KeyT node; int32 port; };
    template <typename KeyT>  struct NodePort { KeyT node; int32 port; };

    // from(), to(), node_port()
    template <typename KeyT>  From<KeyT> from(const KeyT node, int32 port);
    template <typename KeyT>  To<KeyT> to(const KeyT node, int32 port);
    template <typename KeyT>  NodePort<KeyT> node_port(const KeyT node, int32 port);

    // Link, LinkRange
    template <typename KeyT>  struct Link;
    template <typename KeyT>  class LinkRange;

    //
    // PortGraph
    //
    template <typename KeyT>
    class PortGraph
    {
      public:
        friend LinkRange<KeyT>;

        PortGraph() = default;

        void insert(const From<KeyT> from, const To<KeyT> to);
        void insert(const NodePort<KeyT> node_a, const NodePort<KeyT> node_b);

        int32 num_nodes() const;
        bool has_node(const KeyT &from_node);
        int32 count_edges(const KeyT &from_node) const;

        LinkRange<KeyT> from_node(const KeyT &from_node) const;

      protected:
        std::set<KeyT> m_nodes;

        using MapIt = typename
            std::multimap<KeyT, Link<KeyT>>::const_iterator;

        std::multimap<KeyT, Link<KeyT>> m_links;
        std::map<std::pair<KeyT, int32>, MapIt> m_port_link;
        std::map<std::pair<KeyT, KeyT>, MapIt> m_terminal_link;
    };


    //
    // Link
    //
    template <typename KeyT>
    struct Link
    {
      From<KeyT> from = {};
      To<KeyT> to = {};

      int32 name = -1;

      Link(const From<KeyT> from, const To<KeyT> to);
      Link(const From<KeyT> from, const To<KeyT> to, const int32 name);

      Link() = default;
      Link(const Link &) = default;
      Link(Link &&) = default;
      Link & operator=(const Link &) = default;
      Link & operator=(Link &&) = default;
    };


    //
    // LinkIterator
    //
    template <typename KeyT>
    class LinkIterator
    {
      public:
        using MapIt = typename
            std::multimap<KeyT, Link<KeyT>>::const_iterator;
        LinkIterator(MapIt it)                     : i(it) { }
        const Link<KeyT> & operator*() const       { return i->second; }
        bool operator!=(const LinkIterator &that)  { return i != that.i; }
        LinkIterator & operator++()                { ++i; return *this; }
      private:
        MapIt i;
    };


    //
    // LinkRange
    //
    template <typename KeyT>
    class LinkRange
    {
      public:
        using iterator = LinkIterator<KeyT>;

      protected:
        const PortGraph<KeyT> & m_graph;
        iterator m_begin;
        iterator m_end;
        KeyT m_from_node;

      public:
        LinkRange(
            const PortGraph<KeyT> & graph,
            iterator begin,
            iterator end,
            const KeyT &from_node);

        bool has_port(int32 from_port) const;
        const Link<KeyT> & port(int32 from_port) const;
        const Link<KeyT> & to(const KeyT &to_node) const;

        iterator begin() const  { return m_begin; }
        iterator end() const    { return m_end; }
    };

  }//portgraph
}//dray



// =====================
// Implementations
// =====================

namespace dray
{
  namespace portgraph
  {
    // PortGraph::num_nodes()
    template <typename KeyT>
    int32 PortGraph<KeyT>::num_nodes() const
    {
      return m_nodes.size();
    }

    // PortGraph::has_node()
    template <typename KeyT>
    bool PortGraph<KeyT>::has_node(const KeyT &from_node)
    {
      return m_nodes.find(from_node) != m_nodes.end();
    }

    // PortGraph::count_edges()
    template <typename KeyT>
    int32 PortGraph<KeyT>::count_edges(const KeyT &from_node) const
    {
      return m_links.count(from_node);
    }

    // PortGraph::insert()
    template <typename KeyT>
    void PortGraph<KeyT>::insert(const From<KeyT> from, const To<KeyT> to)
    {
      m_nodes.insert(from.node);

      MapIt new_link =
          m_links.insert({from.node, {from, to}});
      m_port_link.insert({{from.node, from.port}, new_link});
      m_terminal_link.insert({{from.node, to.node}, new_link});
    }

    // PortGraph::insert()
    template <typename KeyT>
    void PortGraph<KeyT>::insert(const NodePort<KeyT> node_a, const NodePort<KeyT> node_b)
    {
      this->insert(from(node_a.node, node_a.port), to(node_b.node, node_b.port));
      this->insert(from(node_b.node, node_b.port), to(node_a.node, node_a.port));
    }

    // PortGraph::from_node()
    template <typename KeyT>
    LinkRange<KeyT> PortGraph<KeyT>::from_node(const KeyT &from_node) const
    {
      std::pair<MapIt, MapIt> range = m_links.equal_range(from_node);
      return LinkRange<KeyT>(*this, {range.first}, {range.second}, from_node);
    }


    // Link::Link()
    template <typename KeyT>
    Link<KeyT>::Link(const From<KeyT> from, const To<KeyT> to)
      :
        from(std::move(from)), to(std::move(to))
    { }

    // Link::Link()
    template <typename KeyT>
    Link<KeyT>::Link(
        const From<KeyT> from,
        const To<KeyT> to,
        const int32 name)
      :
        Link(from, to), name(name)
    { }


    // LinkRange::LinkRange()
    template <typename KeyT>
    LinkRange<KeyT>::LinkRange(
        const PortGraph<KeyT> & graph,
        iterator begin,
        iterator end,
        const KeyT &from_node)
      :
        m_graph(graph), m_begin(begin), m_end(end), m_from_node(from_node)
    { }

    // LinkRange::has_port()
    template <typename KeyT>
    bool LinkRange<KeyT>::has_port(int32 from_port) const
    {
      return m_graph.m_port_link.find({m_from_node, from_port}) !=
             m_graph.m_port_link.end();
    }

    // LinkRange::port()
    template <typename KeyT>
    const Link<KeyT> & LinkRange<KeyT>::port(int32 from_port) const
    {
      return m_graph.m_port_link.at({m_from_node, from_port})->second;
    }

    // LinkRange::to()
    template <typename KeyT>
    const Link<KeyT> & LinkRange<KeyT>::to(const KeyT &to_node) const
    {
      return m_graph.m_terminal_link.at({m_from_node, to_node})->second;
    }


    // from()
    template <typename KeyT>
    From<KeyT> from(const KeyT node, int32 port)
    {
      return {std::move(node), port};
    }

    // to()
    template <typename KeyT>
    To<KeyT> to(const KeyT node, int32 port)
    {
      return {std::move(node), port};
    }

    // node_port()
    template <typename KeyT>
    NodePort<KeyT> node_port(const KeyT node, int32 port)
    {
      return {std::move(node), port};
    }

  }//portgraph
}//dray

#endif//DRAY_CPU_GRAPH_HPP

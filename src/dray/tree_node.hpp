// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#ifndef DRAY_TREE_NODE_HPP
#define DRAY_TREE_NODE_HPP

#include <dray/types.hpp>
#include <dray/exports.hpp>
#include <dray/lattice_coordinate.hpp>

namespace dray
{
  template <int32 dim>
  class TreeNode
  {
    private:
      LatticeCoord<dim> m_coord;
      int32 m_level;
      TreeNode();

    public:
      DRAY_EXEC static TreeNode root();

      DRAY_EXEC TreeNode(const LatticeCoord<dim> &coord, int32 level);

      DRAY_EXEC TreeNode(const TreeNode &) = default;
      DRAY_EXEC TreeNode(TreeNode &&) = default;
      DRAY_EXEC TreeNode & operator=(const TreeNode &) = default;
      DRAY_EXEC TreeNode & operator=(TreeNode &&) = default;

      DRAY_EXEC const LatticeCoord<dim> & coord() const;
      DRAY_EXEC void coord(const LatticeCoord<dim> & coord);

      DRAY_EXEC const int32 level() const;
      DRAY_EXEC void level(int32 level);

      DRAY_EXEC TreeNode parent() const;
      DRAY_EXEC TreeNode child(int32 child_num) const;
  };
}


namespace dray
{
  template <int32 dim>
  DRAY_EXEC TreeNode<dim> TreeNode<dim>::root()
  {
    LatticeCoord<dim> coord;
    for (int32 d = 0; d < dim; ++d)
      coord.set(d, 0.f);
    int32 level = 0;
    return TreeNode(coord, level);
  }

  template <int32 dim>
  DRAY_EXEC TreeNode<dim>::TreeNode(const LatticeCoord<dim> &coord, int32 level)
    : m_level(level)
  {
    LatticeCoord<dim> trunc_coord;
    for (int32 d = 0; d < dim; ++d)
      trunc_coord.set(d, trunc(coord.at(d), level+1));
    m_coord = trunc_coord;
  }

  template <int32 dim>
  DRAY_EXEC const LatticeCoord<dim> & TreeNode<dim>::coord() const
  {
    return m_coord;
  }

  template <int32 dim>
  DRAY_EXEC void TreeNode<dim>::coord(const LatticeCoord<dim> & coord)
  {
    LatticeCoord<dim> trunc_coord;
    for (int32 d = 0; d < dim; ++d)
      trunc_coord.set(d, trunc(coord.at(d), level() + 1));
    m_coord = trunc_coord;
  }

  template <int32 dim>
  DRAY_EXEC const int32 TreeNode<dim>::level() const
  {
    return m_level;
  }

  template <int32 dim>
  DRAY_EXEC void TreeNode<dim>::level(int32 level)
  {
    m_level = level;
  }

  template <int32 dim>
  DRAY_EXEC TreeNode<dim> TreeNode<dim>::parent() const
  {
    const int32 parent_level = (level() == 0 ? 0 : level() - 1);
    return TreeNode(coord(), parent_level);
  }

  template <int32 dim>
  DRAY_EXEC TreeNode<dim> TreeNode<dim>::child(int32 child_num) const
  {
    constexpr int32 max_depth = FixedPoint::bits - 1;
    const int32 child_level = (level() == max_depth ? max_depth : level() + 1);
    LatticeCoord<dim> child_coord = coord();
    if (child_level > level())
    {
      for (int32 d = 0; d < dim; ++d)
      {
        FixedPoint coord_d = child_coord.at(d);
        coord_d.digit(child_level, bool(child_num & (1u << d)));
        child_coord.set(d, coord_d);
      }
    }
    return TreeNode(child_coord, child_level);
  }
}

#endif//DRAY_TREE_NODE_HPP

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MERGER_HPP
#define DRAY_MERGER_HPP

#include <assert.h>

/**
 * Mergeable Interface:
 *
 *   Mergeable();                                    // Construct identity.
 *   Mergeable(const Mergeable &other);              // Copy constructor.
 *   void operator=(const Mergeable &other);         // Copy assignment operator.
 *   void operator=(Mergeable &&other);              // Move assignment operator.
 *   void Mergeable::merge(const Mergeable &other);  // Merges two objects in place.
 *   void Mergeable::reset();                        // Resets to the identity object.
 *
 */

template <class Mergeable, int levels = 5>
struct Merger
{
  Mergeable m_merge_nodes[levels];
  unsigned int m_count;

  DRAY_EXEC Merger() : m_count(0u) {}

  DRAY_EXEC void include(const Mergeable &obj, unsigned int rank = 0)
  {
    Mergeable carry(obj);

    assert(rank < levels);
    assert(m_count + (1u << rank) < (1u << levels));  // Else, increase levels.

    while (m_count & (1u << rank))
    {
      carry.merge(m_merge_nodes[rank]);
      m_merge_nodes[rank].reset();
      m_count -= (1u << rank);
      ++rank;
    }
    m_merge_nodes[rank] = std::move(carry);
    m_count += (1u << rank);
  }

  DRAY_EXEC Mergeable final_merge() const
  {
    Mergeable acc;
    for (int l = 0; l < levels && bool(m_count >> l); ++l)
      acc.merge(m_merge_nodes[l]);
    return acc;
  }
};


#endif//DRAY_MERGER_HPP

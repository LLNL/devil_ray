// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TO_BERNSTEIN_HPP
#define DRAY_TO_BERNSTEIN_HPP

#include <dray/data_set.hpp>
#include <dray/derived_topology.hpp>
#include <dray/GridFunction/field.hpp>

namespace dray
{


class ToBernstein
{
  protected:
  public:
    DataSet execute(DataSet &data_set);
};

};//namespace dray


#endif

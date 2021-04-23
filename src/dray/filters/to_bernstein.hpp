// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TO_BERNSTEIN_HPP
#define DRAY_TO_BERNSTEIN_HPP

#include <dray/data_set.hpp>
#include <dray/collection.hpp>
#include <dray/derived_topology.hpp>
#include <dray/data_model/field.hpp>

namespace dray
{


class ToBernstein
{
  protected:
    DataSet execute(DataSet &data_set);
  public:
    Collection execute(Collection &collxn);
};

};//namespace dray


#endif

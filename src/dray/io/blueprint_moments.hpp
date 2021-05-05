// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BLUEPRINT_MOMENTS_HPP
#define DRAY_BLUEPRINT_MOMENTS_HPP

#include <dray/data_model/collection.hpp>
#include <dray/data_model/data_set.hpp>

namespace dray
{

  namespace detail
  {
    // -------------------------------------------------
    class ArrayMapping;   // see dray/io/array_mapping.hpp

    Collection import_into_uniform_moments(const conduit::Node &n_all_domains, ArrayMapping & amap, int32 &num_moments);
    DataSet import_domain_into_uniform_moments(const conduit::Node &n_dataset, ArrayMapping & amap, int32 &num_moments);
    void export_from_uniform_moments(const Collection &dray_collection, ArrayMapping & amap, conduit::Node &n_all_domains);
    void export_from_uniform_moments(const DataSet &dray_dataset, ArrayMapping & amap, conduit::Node &n_dataset);
    // -------------------------------------------------

  }//namespace detail
}//namespace dray

#endif//DRAY_BLUEPRINT_MOMENTS_HPP

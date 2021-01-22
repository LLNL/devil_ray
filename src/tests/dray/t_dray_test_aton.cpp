// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/filters/first_scatter.hpp>

/// #include <dray/io/blueprint_reader.hpp>
#include <dray/io/blueprint_moments.hpp>
#include <dray/io/array_mapping.hpp>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>


TEST(aton_dray, aton_import_and_integrate)
{
  conduit::Node data;
  conduit::relay::io::load("../../../debug/external_source.json", "json", data);

  dray::int32 num_moments;
  dray::detail::ArrayMapping amap;
  dray::Collection dray_collection = dray::detail::import_into_uniform_moments(data, amap, num_moments);

  dray::FirstScatter first_scatter;
  first_scatter.emission_field("first_scatter");
  first_scatter.total_cross_section_field("sigt");
  first_scatter.legendre_order(sqrt(num_moments) - 1);
  first_scatter.uniform_isotropic_scattering(0.05f);  // TODO don't assume uniform scattering

  first_scatter.overwrite_first_scatter_field("first_scatter");
  first_scatter.execute(dray_collection);

  dray::detail::export_from_uniform_moments(dray_collection, amap, data);

  conduit::relay::io::save(data, "first_scatter_source.json");
}



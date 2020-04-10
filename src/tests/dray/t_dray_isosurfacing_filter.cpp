// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <dray/filters/isosurfacing.hpp>

#include <dray/import_order_policy.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <dray/utils/data_logger.hpp>
#include <dray/error.hpp>

TEST (dray_isosurface_filter, dray_isosurface_filter)
{

  std::string root_file = std::string (DATA_DIR) + "taylor_green.cycle_000190.root";

  dray::DataSet dataset = dray::BlueprintReader::load (root_file, dray::ImportOrderPolicy::fixed());

  const float isoval = 0.09;

  std::shared_ptr<dray::ExtractIsosurface> iso_extractor
    = std::make_shared<dray::ExtractIsosurface>();
  iso_extractor->iso_field("velocity_x");
  iso_extractor->iso_value(isoval);

  dray::DataSet isosurf = iso_extractor->execute(dataset);

  { using namespace dray;
    DRAY_INFO("Extracted as surface. Now to render.");
  }

  { using namespace dray;
    DRAY_ERROR("TODO render it");
  }
}

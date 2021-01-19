// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef HRD_LOADER_H
#define HRD_LOADER_H

#include <string>
#include <dray/array.hpp>
#include <dray/vec.hpp>

namespace dray
{

Array<Vec<float32,3>> read_hdr_image(const std::string filename, int &width, int &height);

}

#endif

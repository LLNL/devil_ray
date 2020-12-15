// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#ifndef DRAY_POINT_WRITER_HPP
#define DRAY_POINT_WRITER_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/types.hpp>
#include <dray/exports.hpp>

namespace dray
{

void write_points(Array<Vec<Float,3>> points, const std::string name = "points");

} // namespace dray

#endif//DRAY_POINT_WRITER_HPP

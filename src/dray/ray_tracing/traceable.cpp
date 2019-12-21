// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/ray_tracing/traceable.hpp>

namespace dray
{
namespace ray_tracing
{

Traceable::Traceable(DataSet &data_set)
  : m_data_set(data_set)
{
}

Traceable::~Traceable()
{
}

void Traceable::input(DataSet &data_set)
{
  m_data_set = data_set;
}

}} // namespace dray::ray_tracing

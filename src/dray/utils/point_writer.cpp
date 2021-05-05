// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/utils/point_writer.hpp>
#include <fstream>

namespace dray
{

void write_points(Array<Vec<Float,3>> points, const std::string name)
{
  const int32 size = points.size();
  std::ofstream file;
  file.open(name + ".vtk");
  file<<"# vtk DataFile Version 3.0\n";
  file<<"particles\n";
  file<<"ASCII\n";
  file<<"DATASET UNSTRUCTURED_GRID\n";
  file<<"POINTS "<<size<<" double\n";

  const Vec<Float,3> *points_ptr = points.get_host_ptr_const();
  for(int32 i = 0; i < size; ++i)
  {
    Vec<Float,3> point = points_ptr[i];
    file<<point[0]<<" ";
    file<<point[1]<<" ";
    file<<point[2]<<"\n";
  }

  file<<"CELLS "<<size<<" "<<size * 2<<"\n";
  for(int i = 0; i < size; ++i)
  {
    file<<"1 "<<i<<"\n";
  }

  file<<"CELL_TYPES "<<size<<"\n";
  for(int i = 0; i < size; ++i)
  {
    file<<"1\n";
  }
}

} // namespace dray


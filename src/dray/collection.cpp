// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/collection.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

Collection::Collection()
{

}

Range Collection::range(const std::string field_name)
{
  Range res;

  for(DataSet &dom : m_domains)
  {
    res.include(dom.field(field_name)->range()[0]);
  }

  return res;
}

Range
Collection::global_range(const std::string field_name)
{
  Range res = range(field_name);
#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());

  float64 local_min = (float64) res.min();
  float64 local_max = (float64) res.max();
  float64 global_min = 0;
  float64 global_max = 0;

  MPI_Allreduce((void *)(&local_min),
                (void *)(&global_min),
                1,
                MPI_DOUBLE,
                MPI_MIN,
                mpi_comm);

  MPI_Allreduce((void *)(&local_max),
                (void *)(&global_max),
                1,
                MPI_DOUBLE,
                MPI_MAX,
                mpi_comm);
  res.reset();
  res.include(Float(global_min));
  res.include(Float(global_max));
#endif
  return res;
}

AABB<3>
Collection::bounds()
{
  AABB<3> res;

  for(DataSet &dom : m_domains)
  {
    res.include(dom.topology()->bounds());
  }

  return res;
}

AABB<3>
Collection::global_bounds()
{
  AABB<3> res = bounds();
#ifdef DRAY_MPI_ENABLED

  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  AABB<3> global_bounds;
  for(int i = 0; i < 3; ++i)
  {

    float64 local_min = (float64)res.m_ranges[i].min();
    float64 local_max = (float64)res.m_ranges[i].max();
    float64 global_min = 0;
    float64 global_max = 0;

    MPI_Allreduce((void *)(&local_min),
                  (void *)(&global_min),
                  1,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_comm);

    MPI_Allreduce((void *)(&local_max),
                  (void *)(&global_max),
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  mpi_comm);

    global_bounds.m_ranges[i].include(Float(global_min));
    global_bounds.m_ranges[i].include(Float(global_max));
  }
  res.include(global_bounds);
#endif
  return res;
}

int32
Collection::topo_dims()
{
  int dims = 0;
  for(DataSet &dom : m_domains)
  {
    dims = std::max(dom.topology()->dims(), dims);
  }
#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());

  int global_dims;
  MPI_Allreduce((void *)(&dims),
                (void *)(&global_dims),
                1,
                MPI_INT,
                MPI_MAX,
                mpi_comm);

  dims = global_dims;
#endif
  return dims;
}

int32
Collection::size()
{
  return (int32)m_domains.size();
}

DataSet
Collection::domain(int32 index)
{
  if(index < 0 || index  > size() - 1)
  {
    DRAY_ERROR("Invalid domain index");
  }
  return m_domains[index];
}

} // namespace dray

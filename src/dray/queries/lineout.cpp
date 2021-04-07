#include <dray/queries/lineout.hpp>

#include <dray/error.hpp>
#include <dray/array_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/dray.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif

namespace dray
{

namespace detail
{


bool has_data(const Array<Location> &locs)
{
  const int32 size = locs.size ();

  const Location *locs_ptr = locs.get_device_ptr_const();
  RAJA::ReduceMax<reduce_policy, int32> max_value (-1);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 el_id = locs_ptr[i].m_cell_id;
    max_value.max(el_id);
  });
  DRAY_ERROR_CHECK();

  // if a cell was located then there is valid data here
  return max_value.get() != -1;
}

#ifdef DRAY_MPI_ENABLED
// TODO: just put these functions into a mpi_utils class
void mpi_send(const float32 *data, int32 count, int32 dest, int32 tag, MPI_Comm comm)
{
  MPI_Send(data, count, MPI_FLOAT, dest, tag, comm);
}

void mpi_send(const float64 *data, int32 count, int32 dest, int32 tag, MPI_Comm comm)
{
  MPI_Send(data, count, MPI_DOUBLE, dest, tag, comm);
}

void mpi_recv(float32 *data, int32 count, int32 src, int32 tag, MPI_Comm comm)
{
  MPI_Recv(data, count, MPI_FLOAT, src, tag, comm, MPI_STATUS_IGNORE);
}

void mpi_recv(float64 *data, int32 count, int32 src, int32 tag, MPI_Comm comm)
{
  MPI_Recv(data, count, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
}

void mpi_bcast(float64 *data, int32 count, int32 root, MPI_Comm comm)
{
  MPI_Bcast(data, count, MPI_DOUBLE, root, comm);
}

void mpi_bcast(float32 *data, int32 count, int32 root, MPI_Comm comm)
{
  MPI_Bcast(data, count, MPI_FLOAT, root, comm);
}
#endif

void merge_values(const Float *src, Float *dst, const int32 size, const Float empty_value)
{
  //  we are doing this on the host because we don't want to pay the mem transfer
  //  cost and we are just going to turn around and broadcast the data back to
  //  all ranks
  for(int32 i = 0; i < size; ++i)
  {
    Float value = src[i];
    if(value != empty_value)
    {
      dst[i] = value;
    }
  }
}

void gather_data(std::vector<Array<Float>> &values, bool has_data, const Float empty_value)
{
#ifdef DRAY_MPI_ENABLED
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  int32 has = has_data ? 1 : 0;
  int32 mpi_size = dray::mpi_size();
  int32 mpi_rank = dray::mpi_rank();

  int32 *ranks_data = new int32[mpi_size];

  MPI_Allgather(&has, 1, MPI_INT, ranks_data, 1, MPI_INT, comm);

  // we know we have at least one variable
  const int32 array_size = values[0].size();
  // we also know that we are only doing scalars at the moment
  Float *temp = new Float[array_size];
  const int32 num_vars = values.size();

  // loop through the ranks that actually have line data
  // and gather the data to rank 0
  for(int32 rank = 1; rank < mpi_size; ++rank)
  {
    if(ranks_data[rank] == 1)
    {
      for(int32 i = 0; i < num_vars; ++i)
      {
        if(mpi_rank == 0)
        {
          mpi_recv(temp, array_size, rank, 0, comm);
          merge_values(temp, values[i].get_host_ptr(), array_size, empty_value);
        }
        else
        {
          mpi_send(values[i].get_host_ptr_const(), array_size, 0, 0, comm);
        }

      }
    }
  }
  // now turn around and broadcast the data back to all ranks
  // NOTE: we could make this a parameter, but we might use this
  // within ascent's expressions so all ranks would need the data
  for(int32 i = 0; i < num_vars; ++i)
  {
    mpi_bcast(values[i].get_host_ptr(), array_size, 0, comm);
  }

  delete[] temp;
  delete[] ranks_data;

#endif
}

struct LineoutLocateFunctor
{
  Array<Vec<Float,3>> m_points;
  Array<Vec<Float,3>> m_ref_points;
  LineoutLocateFunctor(const Array<Vec<Float,3>> &points)
    : m_points(points)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    //m_res = detail::reflect_execute(topo.mesh(), m_point, m_normal);
  }
};

}//namespace detail

Lineout::Lineout()
  : m_samples(100),
    m_empty_val(0)
{
}

int32
Lineout::samples() const
{
  return m_samples;
}

void
Lineout::samples(int32 samples)
{
  if(samples < 1)
  {
    DRAY_ERROR("Number of samples must be positive");
  }
  m_samples = samples;
}

void
Lineout::add_line(const Vec<Float,3> start, const Vec<Float,3> end)
{
  m_starts.push_back(start);
  m_ends.push_back(end);
}

void
Lineout::add_var(const std::string var)
{
  m_vars.push_back(var);
}

void
Lineout::empty_val(const Float val)
{
  m_empty_val = val;
}

Array<Vec<Float,3>>
Lineout::create_points()
{
  const int32 lines_size = m_starts.size();
  // number of samples will be at the beginning + end of
  // the line plus how ever many samples the user asks for
  // m_samples must be > 0
  const int32 samples = m_samples + 2;
  const int32 total_points = lines_size * samples;

  Array<Vec<Float,3>> starts;
  Array<Vec<Float,3>> ends;
  starts.resize(lines_size);
  ends.resize(lines_size);
  // pack the points
  {
    Vec<Float,3> *starts_ptr = starts.get_host_ptr();
    Vec<Float,3> *ends_ptr = ends.get_host_ptr();
    for(int32 i = 0; i < lines_size; ++i)
    {
      starts_ptr[i] = m_starts[i];
      ends_ptr[i] = m_ends[i];
    }
  }

  Array<Vec<Float,3>> points;
  points.resize(total_points);
  Vec<Float,3> *points_ptr = points.get_device_ptr();

  const Vec<Float,3> *starts_ptr = starts.get_device_ptr_const();
  const Vec<Float,3> *ends_ptr = ends.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_points), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 line_id = i / samples;
    const int32 sample_id = i % samples;
    const Vec<Float,3> start = starts_ptr[line_id];
    const Vec<Float,3> end = ends_ptr[line_id];
    Vec<Float,3> dir = end - start;
    const Float step = dir.magnitude() / Float(samples - 1);
    dir.normalize();
    Vec<Float,3> point = start + (step * Float(sample_id)) * dir;

    points_ptr[i] = point;
  });
  DRAY_ERROR_CHECK();

  return points;
}

Lineout::Result
Lineout::execute(Collection &collection)
{
  const int32 vars_size = m_vars.size();
  if(vars_size == 0)
  {
    DRAY_ERROR("Lineout: must specify at least 1 variables:");
  }
  const int32 lines_size = m_starts.size();
  if(lines_size == 0)
  {
    DRAY_ERROR("Lineout: must specify at least 1 line:");
  }

  for(int32 i = 0; i < vars_size; ++i)
  {
    if(!collection.has_field(m_vars[i]))
    {
      DRAY_ERROR("Lineout: no variable named '"<<m_vars[i]<<"'");
    }
  }

  AABB<3> bounds = collection.bounds();
  int32 topo_dims = collection.topo_dims();
  if(topo_dims == 2)
  {
    Range z_range = bounds.m_ranges[2];
    if(z_range.length() != 0)
    {
      DRAY_ERROR("Cannot perform lineout on 2d data where z != 0, "<<
                 "i.e., all data must be on a plane.");
    }
  }

  Array<Vec<Float,3>> points = create_points();
  std::vector<Array<Float>> values;
  values.resize(vars_size);
  for(int32 i = 0; i < vars_size; ++i)
  {
    values[i].resize(points.size());
    array_memset(values[i], m_empty_val);
  }

  bool has_data = false;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    // we are looping over all the sample points in each domain.
    // if the points are not found, the values won't be updated,
    // so at the end, we will should have all the field values
    DataSet data_set = collection.domain(i);
    Array<Location> locs = data_set.topology()->locate(points);
    bool domain_has_data = detail::has_data(locs);
    if(domain_has_data)
    {
      has_data = true;
      for(int32 f = 0; f < vars_size; ++f)
      {
        // TODO: one day we might need to check if this
        // particular data has each field
        data_set.field(m_vars[f])->eval(locs, values[f]);
      }
    }
  }

  detail::gather_data(values, has_data, m_empty_val);

  Result res;
  res.m_points = points;
  // start + end + samples
  res.m_points_per_line = m_samples + 2;
  res.m_vars = m_vars;
  res.m_values = values;

  return res;
}


}//namespace dray

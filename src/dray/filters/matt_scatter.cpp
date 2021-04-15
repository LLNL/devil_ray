
#include <dray/filters/matt_scatter.hpp>
#include <dray/error.hpp>
#include <dray/dray.hpp>
#include <dray/policies.hpp>
#include <dray/utils/point_writer.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/array_utils.hpp>
#include <dray/device_array.hpp>
#include <dray/spherical_harmonics.hpp>
#include <dray/Element/elem_utils.hpp>

#include <algorithm>

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif



namespace dray
{

namespace detail
{

// CPU only for comm
struct RayData
{
  int64 m_ray_id;
  Float m_seg_length;
  Float *m_group_optical_depths;
};

struct UniformLocator
{

  Vec<Float,3> m_origin;
  Vec<Float,3> m_spacing;
  Vec<int32,3> m_cell_dims;
  Vec<Float,3> m_max_point;

  DRAY_EXEC
  UniformLocator(const Vec<Float,3> &origin,
                 const Vec<Float,3> &spacing,
                 const Vec<int32,3> &cell_dims)
    : m_origin(origin),
      m_spacing(spacing),
      m_cell_dims(cell_dims)
  {
    Vec<Float,3> unit_length;
    unit_length[0] = static_cast<Float>(cell_dims[0]);
    unit_length[1] = static_cast<Float>(cell_dims[1]);
    unit_length[2] = static_cast<Float>(cell_dims[2]);
    m_max_point[0] = m_origin[0] + spacing[0] * unit_length[0];
    m_max_point[1] = m_origin[1] + spacing[1] * unit_length[1];
    m_max_point[2] = m_origin[2] + spacing[2] * unit_length[2];
  }

  DRAY_EXEC
  Float hit(const Ray &ray) const
  {
    Float dirx = ray.m_dir[0];
    Float diry = ray.m_dir[1];
    Float dirz = ray.m_dir[2];
    Float origx = ray.m_orig[0];
    Float origy = ray.m_orig[1];
    Float origz = ray.m_orig[2];

    const Float inv_dirx = rcp_safe (dirx);
    const Float inv_diry = rcp_safe (diry);
    const Float inv_dirz = rcp_safe (dirz);

    const Float odirx = origx * inv_dirx;
    const Float odiry = origy * inv_diry;
    const Float odirz = origz * inv_dirz;

    const Float xmin = m_origin[0] * inv_dirx - odirx;
    const Float ymin = m_origin[1] * inv_diry - odiry;
    const Float zmin = m_origin[2] * inv_dirz - odirz;
    const Float xmax = m_max_point[0] * inv_dirx - odirx;
    const Float ymax = m_max_point[1] * inv_diry - odiry;
    const Float zmax = m_max_point[2] * inv_dirz - odirz;

    const Float min_int = ray.m_near;
    Float min_dist =
    max (max (max (min (ymin, ymax), min (xmin, xmax)), min (zmin, zmax)), min_int);
    Float max_dist = min (min (max (ymin, ymax), max (xmin, xmax)), max (zmin, zmax));
    max_dist = min(max_dist, ray.m_far);

    Float dist = infinity<Float>();
    if (max_dist > min_dist)
    {
      dist = min_dist;
    }

    return dist;
  }

  DRAY_EXEC
  Vec<int32,3> logical_cell_index(const int32 &cell_id) const
  {
    Vec<int32,3> idx;
    idx[0] = cell_id % (m_cell_dims[0]);
    idx[1] = (cell_id/ (m_cell_dims[0])) % (m_cell_dims[1]);
    idx[2] = cell_id / ((m_cell_dims[0]) * (m_cell_dims[1]));
    return idx;
  }

  DRAY_EXEC
  Vec<Float,3> cell_center(const int32 &cell_id) const
  {
    Vec<int32,3> idx = logical_cell_index(cell_id);
    Vec<Float,3> center;
    center[0] = m_origin[0] + Float(idx[0]) * m_spacing[0] + m_spacing[0] * 0.5f;
    center[1] = m_origin[1] + Float(idx[1]) * m_spacing[1] + m_spacing[1] * 0.5f;
    center[2] = m_origin[2] + Float(idx[2]) * m_spacing[2] + m_spacing[2] * 0.5f;
    return center;
  }


  DRAY_EXEC
  bool is_inside(const Vec<Float,3> &point) const
  {
    bool inside = true;
    if (point[0] < m_origin[0] || point[0] > m_max_point[0])
    {
      inside = false;
    }
    if (point[1] < m_origin[1] || point[1] > m_max_point[1])
    {
      inside = false;
    }
    if (point[2] < m_origin[2] || point[2] > m_max_point[2])
    {
      inside = false;
    }
    return inside;
  }

  // only call if the point is inside
  DRAY_EXEC
  int32 locate(const Vec<Float,3> &point) const
  {
    Vec<Float,3> temp = point;
    temp = temp - m_origin;
    temp[0] /= m_spacing[0];
    temp[1] /= m_spacing[1];
    temp[2] /= m_spacing[2];
    //make sure that if we border the upper edge we return
    // a consistent cell
    if (temp[0] == Float(m_cell_dims[0]))
      temp[0] = Float(m_cell_dims[0] - 1);
    if (temp[1] == Float(m_cell_dims[1]))
      temp[1] = Float(m_cell_dims[1] - 1);
    if (temp[2] == Float(m_cell_dims[2]))
      temp[2] = Float(m_cell_dims[2] - 1);
    Vec<int32,3> cell;
    cell[0] = (int32)temp[0];
    cell[1] = (int32)temp[1];
    cell[2] = (int32)temp[2];

    return (cell[2] * m_cell_dims[1] + cell[1]) * m_cell_dims[0] + cell[0];
  }

};

#ifdef DRAY_MPI_ENABLED
void mpi_send(float32 *data, int32 count, int32 dest, int32 tag, MPI_Comm comm)
{
  MPI_Send(data, count, MPI_FLOAT, dest, tag, comm);
}

void mpi_send(float64 *data, int32 count, int32 dest, int32 tag, MPI_Comm comm)
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

void allgatherv(float32 *local,
                const int32 local_size,
                float32 *global,
                std::vector<int32> &counts,
                MPI_Comm comm)
{
  int32 send_size = local_size;
  std::vector<int32> displ;
  displ.resize(counts.size());

  if(counts.size() > 0)
  {
    displ[0] = 0;
  }
  for(int32 i = 1; i < counts.size(); ++i)
  {
    displ[i] = counts[i-1] + displ[i-1];
  }

  MPI_Allgatherv(local,
                 send_size,
                 MPI_FLOAT,
                 global,
                 &counts[0],
                 &displ[0],
                 MPI_FLOAT,
                 comm);
}

void allgatherv(float64 *local,
                const int32 local_size,
                float64 *global,
                std::vector<int32> &counts,
                MPI_Comm comm)
{
  int32 send_size = local_size;
  std::vector<int32> displ;
  displ.resize(counts.size());

  if(counts.size() > 0)
  {
    displ[0] = 0;
  }
  for(int32 i = 1; i < counts.size(); ++i)
  {
    displ[i] = counts[i-1] + displ[i-1];
  }

  MPI_Allgatherv(local,
                 send_size,
                 MPI_DOUBLE,
                 global,
                 &counts[0],
                 &displ[0],
                 MPI_DOUBLE,
                 comm);
}

void allgatherv(std::vector<float32> &local,
                std::vector<float32> &global,
                std::vector<int32> &counts,
                MPI_Comm comm)
{
  int32 send_size = local.size();
  std::vector<int32> displ;
  displ.resize(counts.size());

  if(counts.size() > 0)
  {
    displ[0] = 0;
  }
  for(int32 i = 1; i < counts.size(); ++i)
  {
    displ[i] = counts[i-1] + displ[i-1];
  }

  MPI_Allgatherv(&local[0],
                 send_size,
                 MPI_FLOAT,
                 &global[0],
                 &counts[0],
                 &displ[0],
                 MPI_FLOAT,
                 comm);
}

void allgatherv(std::vector<float64> &local,
                std::vector<float64> &global,
                std::vector<int32> &counts,
                MPI_Comm comm)
{
  int32 send_size = local.size();
  std::vector<int32> displ;
  displ.resize(counts.size());

  if(counts.size() > 0)
  {
    displ[0] = 0;
  }

  for(int32 i = 1; i < counts.size(); ++i)
  {
    displ[i] = counts[i-1] + displ[i-1];
  }

  MPI_Allgatherv(&local[0],
                 send_size,
                 MPI_DOUBLE,
                 &global[0],
                 &counts[0],
                 &displ[0],
                 MPI_DOUBLE,
                 comm);
}

void allgatherv(std::vector<int32> &local,
                std::vector<int32> &global,
                std::vector<int32> &counts,
                MPI_Comm comm)
{
  int32 send_size = local.size();
  std::vector<int32> displ;
  displ.resize(counts.size());

  if(counts.size() > 0)
  {
    displ[0] = 0;
  }

  for(int32 i = 1; i < counts.size(); ++i)
  {
    displ[i] = counts[i-1] + displ[i-1];
  }

  MPI_Allgatherv(&local[0],
                 send_size,
                 MPI_INT,
                 &global[0],
                 &counts[0],
                 &displ[0],
                 MPI_INT,
                 comm);
}

#endif

// todo: problably can just do this from the beginning, but maybe we only need
// this once
void pack_coords(std::vector<UniformData> &coords,
                 Array<Vec<Float,3>> &origins,
                 Array<Vec<Float,3>> &spacings,
                 Array<Vec<int32,3>> &dims)
{
  const int32 size = coords.size();
  origins.resize(size);
  spacings.resize(size);
  dims.resize(size);

  Vec<Float,3> *orig_ptr = origins.get_host_ptr();
  Vec<Float,3> *space_ptr = spacings.get_host_ptr();
  Vec<int32,3> *dims_ptr = dims.get_host_ptr();
  for(int32 i = 0; i < size; ++i)
  {
    orig_ptr[i] = coords[i].m_origin;
    space_ptr[i] = coords[i].m_spacing;
    dims_ptr[i] = coords[i].m_dims;
  }
}

void pack_ranks(std::vector<UniformData> &coords,
                Array<int32> &ranks)
{
  const int32 size = coords.size();
  ranks.resize(size);

  int32 *rank_ptr = ranks.get_host_ptr();
  for(int32 i = 0; i < size; ++i)
  {
    rank_ptr[i] = coords[i].m_rank;
  }
}


void gather_local_uniform_data(std::vector<DomainData> &dom_data,
                               std::vector<UniformData> &uniform_data)
{
  int32 local_size = dom_data.size();
  uniform_data.resize(local_size);
  for(int32 i = 0; i < local_size; ++i)
  {
    uniform_data[i].m_spacing = dom_data[i].m_topo->spacing();
    uniform_data[i].m_origin = dom_data[i].m_topo->origin();
    uniform_data[i].m_dims = dom_data[i].m_topo->cell_dims();
    uniform_data[i].m_rank = 0;
  }
}

void gather_uniform_data(std::vector<DomainData> &dom_data, std::vector<UniformData> &uniform_data)
{
  int32 local_size = dom_data.size();
#ifdef DRAY_MPI_ENABLED
  int32 rank, procs;

  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &procs);

  std::vector<int32> dom_counts;
  dom_counts.resize(procs);
  MPI_Allgather(&local_size, 1, MPI_INT, &dom_counts[0], 1, MPI_INT, comm);

  int32 global_size = 0;
  for(auto count : dom_counts)
  {
    global_size += count;
  }
  std::cout<<"Global size "<<global_size<<"\n";

  // setup the send and recv buffers
  std::vector<Float> global_float_data;
  global_float_data.resize(global_size * 6);

  std::vector<Float> local_float_data;
  local_float_data.resize(local_size * 6);

  std::vector<int32> global_int_data;
  global_int_data.resize(global_size * 3);

  std::vector<int32> local_int_data;
  local_int_data.resize(local_size * 3);

  for(int32 i = 0; i < local_size; ++i)
  {
    Vec<Float,3> spacing = dom_data[i].m_topo->spacing();
    Vec<Float,3> origin = dom_data[i].m_topo->origin();
    Vec<int32,3> dims = dom_data[i].m_topo->cell_dims();

    local_float_data[i * 6 + 0] = spacing[0];
    local_float_data[i * 6 + 1] = spacing[1];
    local_float_data[i * 6 + 2] = spacing[2];
    local_float_data[i * 6 + 3] = origin[0];
    local_float_data[i * 6 + 4] = origin[1];
    local_float_data[i * 6 + 5] = origin[2];

    local_int_data[i * 3 + 0] = dims[0];
    local_int_data[i * 3 + 1] = dims[1];
    local_int_data[i * 3 + 2] = dims[2];
  }

  std::vector<int32> recv_counts;
  recv_counts.resize(procs);
  for(int32 i = 0; i < procs; ++i)
  {
    recv_counts[i] = dom_counts[i] * 6;
  }
  allgatherv(local_float_data, global_float_data, recv_counts, comm);

  for(int32 i = 0; i < procs; ++i)
  {
    recv_counts[i] = dom_counts[i] * 3;
  }
  allgatherv(local_int_data, global_int_data, recv_counts, comm);

  uniform_data.resize(global_size);
  int32 curr_rank = 0;
  int32 counter = dom_counts[0];
  for(int32 i = 0; i < global_size; ++i)
  {
    uniform_data[i].m_spacing[0] = global_float_data[i * 6 + 0];
    uniform_data[i].m_spacing[1] = global_float_data[i * 6 + 1];
    uniform_data[i].m_spacing[2] = global_float_data[i * 6 + 2];
    uniform_data[i].m_origin[0] = global_float_data[i * 6 + 3];
    uniform_data[i].m_origin[1] = global_float_data[i * 6 + 4];
    uniform_data[i].m_origin[2] = global_float_data[i * 6 + 5];

    uniform_data[i].m_dims[0] = global_int_data[i * 3 + 0];
    uniform_data[i].m_dims[1] = global_int_data[i * 3 + 1];
    uniform_data[i].m_dims[2] = global_int_data[i * 3 + 2];

    if( i >= counter)
    {
      curr_rank++;
      counter += dom_counts[curr_rank];
    }
    uniform_data[i].m_rank = curr_rank;

  }

#else
  gather_local_uniform_data(dom_data, uniform_data);
#endif
}

Array<Vec<Float,3>> gather_sources(std::vector<Array<Vec<Float,3>>> &source_list)
{
  int32 list_size = source_list.size();
  if(list_size == 0)
  {
    //DRAY_ERROR("Empty source list: didn't plan for underloading");
    std::cout<<"Empty source list: didn't plan for underloading. Don't know what a zero count will do (MPI)\n";
  }

  //if(list_size == 1)
  //{
  //  return source_list[0];
  //}

  std::vector<int32> offsets;
  offsets.resize(list_size);

  offsets[0] = 0;
  for(int32 i = 1; i < list_size; ++i)
  {
    offsets[i] = offsets[i-1] + source_list[i-1].size();
  }

  int32 total = offsets[list_size - 1] + source_list[list_size - 1].size();

  Array<Vec<Float,3>> sources;
  sources.resize(total);
  for(int32 i = 0; i < list_size; ++i)
  {
    array_copy(sources, source_list[i], offsets[i]);
  }

#ifdef DRAY_MPI_ENABLED
  int32 rank, procs;

  // vec3 so there are three values per element
  int32 local_size = sources.size() * 3;
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &procs);

  std::vector<int32> counts;
  counts.resize(procs);
  MPI_Allgather(&local_size, 1, MPI_INT, &counts[0], 1, MPI_INT, comm);

  int32 global_size = 0;
  for(auto count : counts)
  {
    global_size += count;
  }
  std::cout<<"Global source size "<<global_size<<"\n";

  Array<Vec<Float,3>> global_sources;
  global_sources.resize(global_size / 3);

  Float *local_ptr = reinterpret_cast<Float*>(sources.get_host_ptr());
  Float *global_ptr = reinterpret_cast<Float*>(global_sources.get_host_ptr());
  allgatherv(local_ptr, local_size, global_ptr, counts, comm);
  sources = global_sources;
#endif
  return sources;
}


inline
void copy_moments(Array<Float> destination_moments,
             int32 _num_moments,
             LowOrderField *uncollided_flux_out)
{
  const int32 zones_times_moments = destination_moments.size();
  const int32 num_moments = num_moments;
  const int32 ngroups = destination_moments.ncomp();

  if (uncollided_flux_out->values().size() != destination_moments.size())
  {
    std::cerr << "Depositing size " << destination_moments.size()
              << " but output has size " << uncollided_flux_out->values().size()
              << "\n";
  }
  if (uncollided_flux_out->values().ncomp() != destination_moments.ncomp())
  {
    std::cerr << "Depositing ncomp " << destination_moments.ncomp()
              << " but output has ncomp " << uncollided_flux_out->values().ncomp()
              << "\n";
  }

  ConstDeviceArray<Float> in_deva(destination_moments);
  NonConstDeviceArray<Float> out_deva(uncollided_flux_out->values());

  // Based on Kripke/Kernel/Scattering.cpp

  RAJA::forall<for_policy> (RAJA::RangeSegment(0, zones_times_moments),
      [=] DRAY_LAMBDA (int32 zone_moment_idx)
  {
    for (int32 group = 0; group < ngroups; ++group)
      out_deva.get_item(zone_moment_idx, group) =
          in_deva.get_item(zone_moment_idx, group);
  });
}

// Assumes that emission uses anisotropic representation,
// i.e. num_items == num_moments * num_zones
// and moments vary faster than zones.
Array<Float> integrate_moments(Array<Ray> &rays,
                               Array<Float> &path_lengths,
                               int32 _legendre_order,
                               Float _cell_volume,
                               const int32 num_cells)
{
  // we need to figure out how many rays per cell we have
  const Ray * ray_ptr = rays.get_device_ptr_const();
  const int32 ray_size = rays.size();
  // we are going to markj the first ray per cell so we can
  // have a single thread integrate all of the moments
  Array<int32> cell_flags;
  cell_flags.resize(ray_size);
  array_memset(cell_flags, 0);
  int32 *flags_ptr = cell_flags.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment(0, ray_size),
      [=] DRAY_LAMBDA (int32 ii)
  {
    int32 cell_id = ray_ptr[ii].m_pixel_id;
    int32 left_id = ii == 0 ?  -1 : ray_ptr[ii-1].m_pixel_id;
    if(cell_id == left_id)
    {
      // I am not am not the one
      return;
    }
    flags_ptr[ii] = 1;

  });

  Array<int32> work_indices = index_flags(cell_flags);
  const int32 *work_indices_ptr = work_indices.get_device_ptr_const();
  const int32 work_size = work_indices.size();

  const int32 num_groups = path_lengths.ncomp();
  const int32 legendre_order = _legendre_order;
  const int32 num_moments = (legendre_order + 1) * (legendre_order + 1);

  Array<Float> destination_moments;
  destination_moments.resize(num_cells * num_moments, num_groups);

  const Float cell_volume = _cell_volume;

  ConstDeviceArray<Float> path_lengths_dev(path_lengths);
  NonConstDeviceArray<Float> destination_moments_dev(destination_moments);

  RAJA::forall<for_policy> (RAJA::RangeSegment(0, work_size),
      [=] DRAY_LAMBDA (int32 ii)
  {
    int32 current_idx = work_indices_ptr[ii];
    // we put the cell id inside pixel_id
    const int32 cell_idx = ray_ptr[current_idx].m_pixel_id;
    // Clear output
    // TODO: just memset the whole thing
    for (int32 nm = 0; nm < num_moments; ++nm)
      for (int32 group = 0; group < num_groups; ++group)
        destination_moments_dev.get_item(num_moments * cell_idx+ nm, group) = 0.0f;

    SphericalHarmonics<Float> sph(legendre_order);

    // iterare through the rays for this cell and eval the spherical harmonics
    bool more_work = true;
    do
    {
      Ray ray = ray_ptr[current_idx];
      const Vec<Float, 3> omega_hat = ray.m_dir;
      const Float rcp_mag2 = rcp_safe(ray.m_far * ray.m_far);
      // Really should use volume-average (over source cell) of rcp_mag2.

      const Float * sph_eval = sph.eval_all(omega_hat);

      for (int32 group = 0; group < num_groups; ++group)
      {
        // this is the amount of energy making it to the cell
        Float dEmission_dV = path_lengths_dev.get_item(current_idx, group);

        const Float trans_source
          = dEmission_dV * cell_volume * rcp_mag2;

        for (int32 nm = 0; nm < num_moments; ++nm)
        {
          const Float spherical_harmonic = sph_eval[nm];
          const Float contribution = spherical_harmonic * trans_source;
          destination_moments_dev.get_item(num_moments * cell_idx + nm, group)
              += contribution;
        }//moments
      }//components

      if(current_idx == ray_size - 1) more_work = false;
      if(more_work)
      {
        // there is only more work if the next ray is the same cell idx.
        // We could restructure this using atomic adds to drop the whole
        // work index stuff
        more_work = cell_idx == ray_ptr[current_idx+1].m_pixel_id;
      }
      if(more_work)
      {
        // ok, its not safe to bump the current indx
        current_idx++;
      }
    }while(more_work);
  });

  return destination_moments;
}


// Assumes that emission uses anisotropic representation,
// i.e. num_items == num_moments * num_zones
// and moments vary faster than zones.
Array<Float> integrate_moments(Array<Vec<Float,3>> &destinations,
                               int32 _legendre_order,
                               Array<Float> &path_lengths,
                               Array<Vec<Float,3>> &ray_sources,
                               Array<int32> &source_cells,
                               Float _cell_volume,
                               LowOrderField *emission)
{
  using sph_t = Float;

  const int32 ncomp = path_lengths.ncomp();
  const int32 legendre_order = _legendre_order;
  const int32 num_moments = (legendre_order + 1) * (legendre_order + 1);
  const int32 num_destinations = destinations.size();
  const int32 num_sources = ray_sources.size();

  Array<Float> destination_moments;
  destination_moments.resize(num_destinations * num_moments, ncomp);

  const Float cell_volume = _cell_volume;

  ConstDeviceArray<Vec<Float, 3>> destinations_dev(destinations);
  ConstDeviceArray<Vec<Float, 3>> ray_sources_dev(ray_sources);
  ConstDeviceArray<Float> path_lengths_dev(path_lengths);
  ConstDeviceArray<int32> source_cells_dev(source_cells);
  ConstDeviceArray<Float> emission_dev(emission->values());

  NonConstDeviceArray<Float> destination_moments_dev(destination_moments);

  RAJA::forall<for_policy> (RAJA::RangeSegment(0, num_destinations),
      [=] DRAY_LAMBDA (int32 dest)
  {
    // Clear output.
    for (int32 nm = 0; nm < num_moments; ++nm)
      for (int32 component = 0; component < ncomp; ++component)
        destination_moments_dev.get_item(num_moments * dest + nm, component) = 0.0f;

    SphericalHarmonics<sph_t> sph(legendre_order);

    // For each source
    //   For each component
    //     For each moment
    //       Multiply-and-accumulate source term with spherical harmonic.
    const Vec<Float, 3> dest_pos = destinations_dev.get_item(dest);
    for (int32 source = 0; source < num_sources; ++source)
    {
      const Vec<Float, 3> omega = (dest_pos - ray_sources_dev.get_item(source));
      const Vec<Float, 3> omega_hat = omega.normalized();
      const Float rcp_mag2 = rcp_safe(omega.magnitude2());
      // Really should use volume-average (over source cell) of rcp_mag2.

      if (omega.magnitude2() == 0.0f)
        continue;

      const sph_t * sph_eval = sph.eval_all(omega_hat);

      const int32 source_idx = source_cells_dev.get_item(source);

      for (int32 component = 0; component < ncomp; ++component)
      {
        // Evaluate emission in the direction of omega_hat.
        Float dEmission_dV = 0.0f;
        for (int32 nm = 0; nm < num_moments; ++nm)
        {
          dEmission_dV += sph_eval[nm]
                         * emission_dev.get_item(num_moments * source_idx + nm,
                                                 component);
        }

        const Float source_dL_dOmega
          = dEmission_dV * cell_volume * rcp_mag2;

        const Float transmitted = path_lengths_dev.get_item(
            num_sources * dest + source, component);

        const Float trans_source = transmitted * source_dL_dOmega;

        for (int32 nm = 0; nm < num_moments; ++nm)
        {
          const sph_t spherical_harmonic = sph_eval[nm];
          const Float contribution = spherical_harmonic * trans_source;
          destination_moments_dev.get_item(num_moments * dest + nm, component)
              += contribution;
        }//moments
      }//components
    }//sources
  });//destinations

  return destination_moments;
}

// Returns cell center for ever cell in topo.
static Array<Vec<Float,3>> cell_centers(UniformTopology &topo)
{

  const Vec<int32,3> cell_dims = topo.cell_dims();
  const Vec<Float,3> origin = topo.origin();
  const Vec<Float,3> spacing = topo.spacing();

  const int32 num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];

  Array<Vec<Float,3>> locations;
  locations.resize(num_cells);
  Vec<Float,3> *loc_ptr = locations.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_cells), [=] DRAY_LAMBDA (int32 index)
  {
    Vec<int32,3> cell_id;
    cell_id[0] = index % cell_dims[0];
    cell_id[1] = (index / cell_dims[0]) % cell_dims[1];
    cell_id[2] = index / (cell_dims[0] * cell_dims[1]);

    Vec<Float,3> loc;
    for(int32 i = 0; i < 3; ++i)
    {
      loc[i] = origin[i] + Float(cell_id[i]) * spacing[i] + spacing[i] * 0.5f;
    }

    loc_ptr[index] = loc;
  });

  return locations;
}

// Returns cell center of every cell in topo
// that has at least one nonzero emission value.
// The nonzero_list array (same length as return)
// contains the original indices of the nonzero cells.
static Array<Vec<Float,3>> cell_centers_nonzero(UniformTopology &topo,
                                                LowOrderField * emission,
                                                int32 _num_moments,
                                                Array<int32> &nonzero_list)
{

  const Vec<int32,3> cell_dims = topo.cell_dims();
  const Vec<Float,3> origin = topo.origin();
  const Vec<Float,3> spacing = topo.spacing();
  const int32 num_moments = _num_moments;

  // Include a zone if any component _of_any_moment_ is nonzero.
  Array<int32> nonzero_moments_list = index_any_nonzero(emission->values());
  const int32 num_nonzero_items = nonzero_moments_list.size();
  Array<int32> uniq_flags;
  uniq_flags.resize (num_nonzero_items);
  // At first, assume all unique.
  array_memset (uniq_flags, 1);
  NonConstDeviceArray<int32> nzm_deva(nonzero_moments_list);
  NonConstDeviceArray<int32> uniq_flags_deva(uniq_flags);
  RAJA::forall<for_policy>(RAJA::RangeSegment(1, num_nonzero_items),
      [=] DRAY_LAMBDA (int32 nzm_index)
  {
    const int32 left_index = nzm_index - 1;
    const int32 zone = nzm_deva.get_item(nzm_index) / num_moments;
    const int32 left_zone = nzm_deva.get_item(left_index) / num_moments;
    if (zone == left_zone)
    {
      uniq_flags_deva.get_item(nzm_index) = 0;
    }
  });

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_nonzero_items),
      [=] DRAY_LAMBDA (int32 nzm_index)
  {
    const int32 zone = nzm_deva.get_item(nzm_index) / num_moments;
    nzm_deva.get_item(nzm_index) = zone;
  });

  nonzero_list = index_flags(uniq_flags, nonzero_moments_list);

  ConstDeviceArray<int32> nonzero_list_deva(nonzero_list);
  const int32 num_nonzero_cells = nonzero_list.size();

  Array<Vec<Float,3>> locations;
  locations.resize(num_nonzero_cells);
  Vec<Float,3> *loc_ptr = locations.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_nonzero_cells),
      [=] DRAY_LAMBDA (int32 nz_index)
  {
    const int32 index = nonzero_list_deva.get_item(nz_index);
    Vec<int32,3> cell_id;
    cell_id[0] = index % cell_dims[0];
    cell_id[1] = (index / cell_dims[0]) % cell_dims[1];
    cell_id[2] = index / (cell_dims[0] * cell_dims[1]);

    Vec<Float,3> loc;
    for(int32 i = 0; i < 3; ++i)
    {
      loc[i] = origin[i] + Float(cell_id[i]) * spacing[i] + spacing[i] * 0.5f;
    }

    loc_ptr[nz_index] = loc;
  });

  return locations;
}

struct FS_TraversalState
{
  Vec<Float,3> m_delta_max;
  Vec<Float,3> m_delta;
  Vec<int32,3> m_voxel;
  Vec<Float,3> m_dir;

  // distance to voxel exit from initial point
  DRAY_EXEC
  Float exit() const
  {
    return min(m_delta_max[0], min(m_delta_max[1], m_delta_max[2]));
  }

  // advances to the next voxel along the ray
  DRAY_EXEC void advance()
  {
    int32 advance_dir = 0;
    for(int32 i = 1; i < 3; ++i)
    {
      if(m_delta_max[i] < m_delta_max[advance_dir])
      {
        advance_dir = i;
      }
    }
    m_delta_max[advance_dir] += m_delta[advance_dir];
    m_voxel[advance_dir] += m_dir[advance_dir] < 0.f ? -1 : 1;
  }

};

struct FS_DDATraversal
{
  const Vec<int32,3> m_dims;
  const Vec<Float,3> m_origin;
  const Vec<Float,3> m_spacing;

  FS_DDATraversal(UniformTopology &topo)
    : m_dims(topo.cell_dims()),
      m_origin(topo.origin()),
      m_spacing(topo.spacing())
  {

  }

  FS_DDATraversal(const Vec<int32,3> &dims,
                  const Vec<Float,3> &origin,
                  const Vec<Float,3> &spacing)
    : m_dims(dims),
      m_origin(origin),
      m_spacing(spacing)
  {

  }

  DRAY_EXEC
  bool is_inside(const Vec<int32, 3>& index) const
  {
    bool inside = true;
    const int32 minIndex = min(index[0], min(index[1], index[2]));
    if(minIndex < 0) inside = false;
    if(index[0] >= m_dims[0]) inside = false;
    if(index[1] >= m_dims[1]) inside = false;
    if(index[2] >= m_dims[2]) inside = false;
    return inside;
  }

  DRAY_EXEC
  int32 voxel_index(const Vec<int32, 3> &voxel) const
  {
    return voxel[0] + voxel[1] * m_dims[0] + voxel[2] * m_dims[0] * m_dims[1];
  }

  DRAY_EXEC Float
  init_traversal(const Vec<Float,3> &point,
                 const Vec<Float,3> &dir,
                 FS_TraversalState &state) const
  {
    Vec<Float, 3> temp = point;
    temp = temp - m_origin;
    state.m_voxel[0] = temp[0] / m_spacing[0];
    state.m_voxel[1] = temp[1] / m_spacing[1];
    state.m_voxel[2] = temp[2] / m_spacing[2];
    state.m_dir = dir;

    Vec<Float,3> step;
    step[0] = (dir[0] >= 0.f) ? 1.f : -1.f;
    step[1] = (dir[1] >= 0.f) ? 1.f : -1.f;
    step[2] = (dir[2] >= 0.f) ? 1.f : -1.f;

    Vec<Float,3> next_boundary;
    next_boundary[0] = (Float(state.m_voxel[0]) + step[0]) * m_spacing[0];
    next_boundary[1] = (Float(state.m_voxel[1]) + step[1]) * m_spacing[1];
    next_boundary[2] = (Float(state.m_voxel[2]) + step[2]) * m_spacing[2];

    // correct next boundary for negative directions
    if(step[0] == -1.f) next_boundary[0] += m_spacing[0];
    if(step[1] == -1.f) next_boundary[1] += m_spacing[1];
    if(step[2] == -1.f) next_boundary[2] += m_spacing[2];

    // distance to next voxel boundary
    state.m_delta_max[0] = (dir[0] != 0.f) ?
      (next_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

    state.m_delta_max[1] = (dir[1] != 0.f) ?
      (next_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

    state.m_delta_max[2] = (dir[2] != 0.f) ?
      (next_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

    // distance along ray to traverse x,y, and z of a voxel
    state.m_delta[0] = (dir[0] != 0) ? m_spacing[0] / dir[0] * step[0] : infinity<Float>();
    state.m_delta[1] = (dir[1] != 0) ? m_spacing[1] / dir[1] * step[1] : infinity<Float>();
    state.m_delta[2] = (dir[2] != 0) ? m_spacing[2] / dir[2] * step[2] : infinity<Float>();

    Vec<Float,3> exit_boundary;
    exit_boundary[0] = step[0] < 0.f ? 0.f : Float(m_dims[0]) * m_spacing[0];
    exit_boundary[1] = step[1] < 0.f ? 0.f : Float(m_dims[1]) * m_spacing[1];
    exit_boundary[2] = step[2] < 0.f ? 0.f : Float(m_dims[2]) * m_spacing[2];

    // Masado questions these lines
    if(step[0] == -1.f) exit_boundary[0] += m_spacing[0];
    if(step[1] == -1.f) exit_boundary[1] += m_spacing[1];
    if(step[2] == -1.f) exit_boundary[2] += m_spacing[2];

    Vec<Float,3> exit_dist;
    // distance to grid exit
    exit_dist[0] = (dir[0] != 0.f) ?
      (exit_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

    exit_dist[1] = (dir[1] != 0.f) ?
      (exit_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

    exit_dist[2] = (dir[2] != 0.f) ?
      (exit_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

    //std::cout<<"Init voxel "<<voxel<<"\n";

    return min(exit_dist[0], min(exit_dist[1], exit_dist[2]));
  }



};


} // namespace detail

UncollidedFlux::UncollidedFlux()
  : m_legendre_order(0),
    m_num_groups(0),
    m_sigs(0.0f)
{

}


// Returns flattened array of size num_destinations * num_sources.
// Results for each source and destination, with sources varying faster.
// Assumes that absorption is isotropic (no dependence on moments).
static
Array<Float>
go_trace(Array<Vec<Float,3>> &destinations,
         Array<Vec<Float,3>> &ray_sources,
         UniformTopology &topo,
         LowOrderField *absorption)
{
  // input
  const detail::FS_DDATraversal dda(topo);
  const Vec<Float,3> *destn_ptr = destinations.get_device_ptr_const();
  const Vec<Float,3> *ray_src_ptr = ray_sources.get_device_ptr_const();
  const int32 size_ray_srcs = ray_sources.size();
  const int32 size = destinations.size();
  const ConstDeviceArray<Float> absorption_arr( absorption->values() );

  const int32 ncomp = absorption_arr.ncomp();

  // output
  Array<Float> path_lengths;
  path_lengths.resize(size * size_ray_srcs, ncomp);
  NonConstDeviceArray<Float> length_arr( path_lengths );

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 index)
  {
    Float * res = new Float[ncomp];

    Vec<Float,3> destn = destn_ptr[index];
    for(int ray_src = 0; ray_src < size_ray_srcs; ++ray_src)
    {
      Vec<Float,3> loc = ray_src_ptr[ray_src];
      Vec<Float,3> dir = destn - loc;
      if (dir.magnitude2() == 0.0)
        continue;

      dir.normalize();
      detail::FS_TraversalState state;
      dda.init_traversal(loc, dir, state);

      Float distance = 0.f;
      for (int32 component = 0; component < ncomp; ++component)
        res[component] = 1.f;

        // while not at end point
      while(dda.is_inside(state.m_voxel))
      {
        const Float voxel_exit = state.exit();
        const Float length = voxel_exit - distance;

        const int32 cell_id = dda.voxel_index(state.m_voxel);
        for (int32 component = 0; component < ncomp; ++component)
        {
          const Float absorb
              = exp(-absorption_arr.get_item(cell_id, component) * length);

          res[component] = res[component] * absorb;
        }
        // this will get more complicated with MPI and messed up
        // metis domain decompositions

        distance = voxel_exit;
        state.advance();
      }
      // Directions matter.
      // Instead of summing over all sources,
      // return result from each source separately.
      for (int32 component = 0; component < ncomp; ++component)
      {
        length_arr.get_item(size_ray_srcs * index + ray_src, component)
            = res[component];
      }
    }

    delete [] res;
  });
  return path_lengths;
}

struct SOASort
{
  Float *m_depths;
  int32 *m_pixel_ids;
  SOASort(Float *depths, int32 *pixel_ids)
    : m_depths(depths),
      m_pixel_ids(pixel_ids)
  {
  }

  bool operator()(int32 i, int32 j)
  {
    if(m_pixel_ids[i] < m_pixel_ids[j])
    {
       return true;
    }
    else if(m_pixel_ids[i] == m_pixel_ids[j])
    {
      return m_depths[i] < m_depths[j];
    }

    return false;
  }
};

void
UncollidedFlux::exchange(std::vector<Array<Ray>> &rays,
                         std::vector<Array<Float>> &ray_data,
                         Array<int32> &res_pixel_ids,
                         Array<Float> &res_path_lengths)
{
  // Technically, we don't need the depths, since
  // we don't care about the order (emission)
  // but i'll keep them anyway for debugging
  Array<int32> out_pixel_ids;
  Array<Float> out_depths;
  Array<Float> out_data;
  const int32 num_groups = m_num_groups;
#ifdef DRAY_MPI_ENABLED
  std::vector<Array<int32>> dests = destinations(rays);

  int32 rank, procs;
  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &procs);

  // fighting the urge to optimize
  std::vector<int32> send_map;
  send_map.resize(procs);
  const int32 local_size = dests.size();
  for(int32 i = 0; i < local_size; ++i)
  {
    int32 * dests_ptr = dests[i].get_host_ptr();
    const int32 dsize = dests[i].size();
    for(int32 ii = 0; ii < dsize; ++ii)
    {
      send_map[dests_ptr[ii]]++;
    }
  }



  std::vector<int32> recv_map, recv_buffer;
  recv_map.resize(procs);
  recv_buffer.resize(procs);

  for(int32 i = 0; i < procs; ++i)
  {
    int32 *buffer = &recv_buffer[0];
    if(rank == i)
    {
      buffer = &send_map[0];
    }

    MPI_Bcast(buffer, procs, MPI_INT, i, comm);
    // we only care about the number of things rank i is
    // sending to us.
    recv_map[i] = buffer[rank];
  }


  // figure out the recvs
  int32 total_incoming = 0;
  std::vector<int32> offsets;
  offsets.resize(procs);
  offsets[0] = 0;
  for(int32 p = 0; p < procs; p++)
  {
    if(p > 0) offsets[p] = offsets[p-1] + recv_map[p-1];
    total_incoming += recv_map[p];
  }

  if(rank == 2)
  {
    std::cout<<"Rank 2 recv map : ";
    for(int32 p = 0; p < procs; p++)
    {
      std::cout<<recv_map[p]<<" ";
    }
    std::cout<<"\n";
  }

  if(ray_data.size() == 0)
  {
    DRAY_ERROR("did not plan for no domains");
  }

  out_pixel_ids.resize(total_incoming);
  out_depths.resize(total_incoming);
  out_data.resize(num_groups * total_incoming);

  std::vector<int32> send_pixel_ids;
  std::vector<Float> send_depths;
  std::vector<Float> send_data;

  // create a waterfall comunication pattern between all snd/rcv pairs
  for(int32 p = 0; p < procs; p++)
  {
    // I have sends or rcvs
    if(send_map[p] > 0 || recv_map[p] > 0)
    {
      // this include packing data destined for this rank
      if(send_map[p] > 0)
      {
        send_pixel_ids.resize(send_map[p]);
        send_depths.resize(send_map[p]);
        send_data.resize(num_groups * send_map[p]);
        //  just go through the buffers and get the data
        int32 counter = 0;
        for(int32 i = 0; i < local_size; ++i)
        {
          const int32 * dests_ptr = dests[i].get_host_ptr_const();
          const Ray * ray_ptr = rays[i].get_host_ptr_const();
          const Float * data_ptr = ray_data[i].get_host_ptr_const();
          const int32 dsize = dests[i].size();
          for(int32 ii = 0; ii < dsize; ++ii)
          {
            int32 d = dests_ptr[ii];
            if(d == p)
            {
              send_depths[counter] = ray_ptr[ii].m_near;
              send_pixel_ids[counter] = ray_ptr[ii].m_pixel_id;
              size_t src_offset = ii * num_groups;
              size_t dst_offset = counter * num_groups;
              memcpy(&send_data[0] + dst_offset, data_ptr + src_offset, sizeof(Float) * num_groups);
              counter++;
            }
          } // ii
        } // i
      } // packing sends

      // this is stuff we are sending to ourself
      if(p == rank && recv_map[p] > 0)
      {
        size_t offset = offsets[p];
        // pass the data through
        std::copy(send_pixel_ids.begin(), send_pixel_ids.end(), out_pixel_ids.get_host_ptr() + offset);
        std::copy(send_depths.begin(), send_depths.end(), out_depths.get_host_ptr() + offset);
        std::copy(send_data.begin(), send_data.end(), out_data.get_host_ptr() + offset * num_groups);
        if(rank == 2)
        {
          std::cout<<"Rank 2 pass through "<< out_pixel_ids.size()<<" into offset "<<offset * num_groups<<"\n";
        }
        continue;
      }

      bool boss = rank < p;
      if(boss)
      {
        if(send_map[p] > 0)
        {
          // helpers for Float
          detail::mpi_send(&send_depths[0], send_depths.size(), p, 0, comm);
          detail::mpi_send(&send_data[0], send_data.size(), p, 1, comm);
          MPI_Send(&send_pixel_ids[0], send_pixel_ids.size(), MPI_INT, p, 2, comm);

        }
        if(recv_map[p] > 0)
        {
          size_t offset = offsets[p];
          int32 count = recv_map[p];
          // recv directly in mega buffers
          detail::mpi_recv(out_depths.get_host_ptr() + offset, count, p, 0, comm);
          detail::mpi_recv(out_data.get_host_ptr() + offset * num_groups, count * num_groups, p, 1, comm);
          MPI_Recv(out_pixel_ids.get_host_ptr() + offset, count, MPI_INT, p, 2, comm, MPI_STATUS_IGNORE);
          if(rank == 2)
          {
            std::cout<<"Rank 2 revc from "<<p<<" count "<<count<<" into offset "<<offset * num_groups<<"\n";
          }
        }
      }
      else
      {
        if(recv_map[p] > 0)
        {
          size_t offset = offsets[p];
          int32 count = recv_map[p];
          // recv directly in mega buffers
          detail::mpi_recv(out_depths.get_host_ptr() + offset, count, p, 0, comm);
          detail::mpi_recv(out_data.get_host_ptr() + offset * num_groups, count * num_groups, p, 1, comm);
          MPI_Recv(out_pixel_ids.get_host_ptr() + offset, count, MPI_INT, p, 2, comm, MPI_STATUS_IGNORE);
          if(rank == 2)
          {
            std::cout<<"Rank 2 revc from "<<p<<" count "<<count<<" into offset "<<offset * num_groups<<"\n";
          }

        }
        if(send_map[p] > 0)
        {
          // helpers for Float
          detail::mpi_send(&send_depths[0], send_depths.size(), p, 0, comm);
          detail::mpi_send(&send_data[0], send_data.size(), p, 1, comm);
          MPI_Send(&send_pixel_ids[0], send_pixel_ids.size(), MPI_INT, p, 2, comm);
        }
      }

    } // if we have anything to send/recv
  } // p

  std::cout<<"Rank = "<<rank<<" sitting on "<<out_depths.size()<<"\n";
#else
  // just add everthing together
  int32 total_size = 0;
  for(int32 i = 0; i < rays.size(); ++i)
  {
    total_size += rays[i].size();
  }

  out_pixel_ids.resize(total_size);
  out_depths.resize(total_size);
  out_data.resize(total_size);


  for(int32 i = 0; i < rays.size(); ++i)
  {
    const Ray * ray_ptr = rays[i].get_host_ptr_const();
    const Float * data_ptr = ray_data[i].get_host_ptr_const();
    const int32 lsize = rays[i].size();
    int32 counter = 0;
    for(int32 ii = 0; ii < lsize; ++ii)
    {
      out_depths.get_host_ptr()[counter] = ray_ptr[ii].m_near;
      out_pixel_ids.get_host_ptr()[counter] = ray_ptr[ii].m_pixel_id;
      size_t src_offset = ii * num_groups;
      size_t dst_offset = counter * num_groups;
      memcpy(out_data.get_host_ptr() + dst_offset, data_ptr + src_offset, sizeof(Float) * num_groups);
    } // ii
  } // i
#endif


  Array<int32> indexes =  array_counting(out_depths.size(), 0, 1);

 // raja doesn't support structs for sorts, so we would need
  // something like this if we were sorting by depth as well
  // as pixel id
 //SOASort sorter(&out_depths[0],&out_pixel_ids[0]);
 //std::sort(indexes_ptr, indexes_ptr + out_depths.size(), sorter);


  // sort the pixel ids and get thier original positions inside indexes
  const int32 out_size = out_pixel_ids.size();
  int32 *pid_ptr = out_pixel_ids.get_device_ptr();
  RAJA::stable_sort_pairs<for_policy>(pid_ptr,
                                      pid_ptr + out_size,
                                      indexes.get_device_ptr(),
                                      RAJA::operators::less<int32>{});

  //if(dray::mpi_rank() == 2)
  //{
  //  int32 *indexes_ptr = indexes.get_host_ptr();
  //  for(int i = 0; i < out_depths.size(); ++i)
  //  {
  //   int32 idx = indexes_ptr[i];
  //   //std::cout<<i<<" pid "<<out_pixel_ids.get_value(i)<<"\n";
  //   //std::cout<<"pid "<<out_pixel_ids.get_value(i)<<" "<<out_data.get_value(idx*m_num_groups)<<" index "<<idx*m_num_groups<<"\n";
  //  }
  //}

  //composite in place
  Array<Float> path_lengths;
  path_lengths.resize(out_size, num_groups);

  Float *plength_ptr = path_lengths.get_device_ptr();
  // this should still be valid, but whatever
  pid_ptr = out_pixel_ids.get_device_ptr();
  int32 *indexes_ptr = indexes.get_device_ptr();

  // these are not sorted
  const Float *data_ptr = out_data.get_device_ptr_const();

  // we will need to compact the output so we mar unique flags here
  Array<int32> compact_flags;
  compact_flags.resize(out_size);
  int32 *flags_ptr = compact_flags.get_device_ptr();

  // this is terribly inefficient but efficiency is not the goal,
  // working solution is the goal
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, out_size), [=] DRAY_LAMBDA (int32 ii)
  {
    flags_ptr[ii] = 0;
    const int32 p_offset = ii * num_groups;
    int32 curr_idx = ii;
    const int32 pid = pid_ptr[curr_idx];
    int32 left_pid = curr_idx == 0 ?  -1 : pid_ptr[curr_idx - 1];
    if(pid == left_pid)
    {
      // I am not alone and I am not in charge
      return;
    }

    // we are are the output
    flags_ptr[ii] = 1;

    // init the memory
    for(int32 group = 0; group < num_groups; ++group)
    {
      plength_ptr[p_offset + group] = 1;
    }

    //int count = 0;
    while(true)
    {
      const int32 orig_idx = indexes_ptr[curr_idx];
      if(pid == 0) std::cout<<"DEBUG  compositing pid "<<pid_ptr[curr_idx]<<"\n";
      for(int32 group = 0; group < num_groups; ++group)
      {
        plength_ptr[p_offset + group] *= data_ptr[orig_idx * num_groups + group];
      }
      curr_idx++;
      if(curr_idx == out_size) break;
      if(pid_ptr[curr_idx] != pid) break;
      //count++;
    }

    //std::cout<<"p_offset "<<p_offset<<" count "<<count<<"\n";

  });

  Array<int32> gather_ids = index_flags(compact_flags);
  std::cout<<"Result size "<<gather_ids.size()<<" from "<<out_size<<"\n";

  const int32 gather_size = gather_ids.size();

  res_path_lengths.resize(gather_size, num_groups);
  res_pixel_ids.resize(gather_size);

  Float *res_lengths_ptr = res_path_lengths.get_device_ptr();
  int32 *res_ids_ptr = res_pixel_ids.get_device_ptr();
  const int32 *gather_ptr = gather_ids.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, gather_size), [=] DRAY_LAMBDA (int32 ii)
  {
    const int32 sparse_idx = gather_ptr[ii];
    res_ids_ptr[ii] = pid_ptr[sparse_idx];

    const int32 in_offset = sparse_idx * num_groups;
    const int32 out_offset = ii * num_groups;
    for(int32 group = 0; group < num_groups; ++group)
    {
      res_lengths_ptr[out_offset + group] = plength_ptr[in_offset + group];
    }

  });

}

std::vector<Array<int32>>
UncollidedFlux::destinations(std::vector<Array<Ray>> &rays)
{

  const int32 global_size = m_global_coords.size();
  const int32 local_size = rays.size();

  Array<Vec<Float,3>> origins;
  Array<Vec<Float,3>> spacings;
  Array<Vec<int32,3>> dims;
  detail::pack_coords(m_global_coords, origins, spacings, dims);

  Array<int32> ranks;
  detail::pack_ranks(m_global_coords, ranks);

  const Vec<Float,3> *origin_ptr = origins.get_device_ptr_const();
  const Vec<Float,3> *space_ptr = spacings.get_device_ptr_const();
  const Vec<int32,3> *dims_ptr = dims.get_device_ptr_const();
  const int32 *rank_ptr = ranks.get_device_ptr_const();

  std::vector<Array<int32>> destinations;

  for(int32 i = 0; i < local_size; ++i)
  {

    const Ray *ray_ptr = rays[i].get_device_ptr_const();
    const int32 num_rays = rays[i].size();
    Array<int32> local_dests;
    local_dests.resize(num_rays);
    int32 * local_dest_ptr = local_dests.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_rays), [=] DRAY_LAMBDA (int32 ii)
    {
      const Ray ray = ray_ptr[ii];
      const Vec<Float,3> point = ray.m_orig + ray.m_dir * ray.m_far;
      int32 rank = -1;

      for(int32 i = 0; i < global_size; ++i)
      {
        detail::UniformLocator locator(origin_ptr[i],
                                       space_ptr[i],
                                       dims_ptr[i]);
        if(locator.is_inside(point))
        {
          rank = rank_ptr[i];
          break;
        }
      }
      if(rank == -1)
      {
        printf("rank bad. this should not happen\n");
      }
      local_dest_ptr[ii] = rank;
      //std::cout<<"Ray "<<ray.m_pixel_id<<" dest "<<rank<<"\n";

    });
    destinations.push_back(local_dests);
  }
  return destinations;
}

void
UncollidedFlux::go_bananas(std::vector<Array<Ray>> &rays,
                           std::vector<Array<Float>> &ray_data)
{

  const int32 local_size = m_domain_data.size();

  for(int32 i = 0; i < local_size; ++i)
  {
    const Vec<int32,3> dims = m_domain_data[i].m_topo->cell_dims();
    const Vec<Float,3> origin = m_domain_data[i].m_topo->origin();
    const Vec<Float,3> spacing = m_domain_data[i].m_topo->spacing();
    const detail::FS_DDATraversal dda(dims,origin,spacing);
    Ray *ray_ptr = rays[i].get_device_ptr();
    const int32 num_rays = rays[i].size();
    const int32 num_groups = m_num_groups;
    const ConstDeviceArray<Float>
      absorption_arr( m_domain_data[i].m_cross_section->values() );

    NonConstDeviceArray<Float> local_data(ray_data[i]);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_rays), [=] DRAY_LAMBDA (int32 ii)
    {
      Ray ray = ray_ptr[ii];
      // where is the ray in this mesh
      Vec<Float,3> start;
      detail::UniformLocator locator(origin,spacing,dims);
      Float near;
      if(locator.is_inside(ray.m_orig))
      {
        start = ray.m_orig;
        near = 0.f;
      }
      else
      {
        // this ray shouldn't exist if it doesn't hit
        near = locator.hit(ray);
        // TODO: make this a relative epsilon
        start = ray.m_orig + ray.m_dir * (near+ 0.0001f);
      }
      // This is pretty much a hack to get some information back that
      // we need for compositing.
      ray.m_near = near;
      ray_ptr[ii] = ray;



      detail::FS_TraversalState state;
      dda.init_traversal(start, ray.m_dir, state);

      Float distance = 0.f;
      while(dda.is_inside(state.m_voxel))
      {
        const Float voxel_exit = state.exit();
        const Float length = voxel_exit - distance;

        const int32 cell_id = dda.voxel_index(state.m_voxel);
        for (int32 group = 0; group < num_groups; ++group)
        {
          const Float absorb
              = exp(-absorption_arr.get_item(cell_id, group) * length);

          local_data.get_item(ii,group) *= absorb;
        }
        // this will get more complicated with MPI and messed up
        // metis domain decompositions

        distance = voxel_exit;
        state.advance();
      }

    });

  }

//  // input
//  const Vec<Float,3> *destn_ptr = destinations.get_device_ptr_const();
//  const Vec<Float,3> *ray_src_ptr = ray_sources.get_device_ptr_const();
//  const int32 size_ray_srcs = ray_sources.size();
//  const int32 size = destinations.size();
//  const ConstDeviceArray<Float> absorption_arr( absorption->values() );
//
//  const int32 ncomp = absorption_arr.ncomp();
//
//  // output
//  Array<Float> path_lengths;
//  path_lengths.resize(size * size_ray_srcs, ncomp);
//  NonConstDeviceArray<Float> length_arr( path_lengths );
//
//  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 index)
//  {
//    Float * res = new Float[ncomp];
//
//    Vec<Float,3> destn = destn_ptr[index];
//    for(int ray_src = 0; ray_src < size_ray_srcs; ++ray_src)
//    {
//      Vec<Float,3> loc = ray_src_ptr[ray_src];
//      Vec<Float,3> dir = destn - loc;
//      if (dir.magnitude2() == 0.0)
//        continue;
//
//      dir.normalize();
//      detail::FS_TraversalState state;
//      dda.init_traversal(loc, dir, state);
//
//      Float distance = 0.f;
//      for (int32 component = 0; component < ncomp; ++component)
//        res[component] = 1.f;
//
//      while(dda.is_inside(state.m_voxel))
//      {
//        const Float voxel_exit = state.exit();
//        const Float length = voxel_exit - distance;
//
//        const int32 cell_id = dda.voxel_index(state.m_voxel);
//        for (int32 component = 0; component < ncomp; ++component)
//        {
//          const Float absorb
//              = exp(-absorption_arr.get_item(cell_id, component) * length);
//
//          res[component] = res[component] * absorb;
//        }
//        // this will get more complicated with MPI and messed up
//        // metis domain decompositions
//
//        distance = voxel_exit;
//        state.advance();
//      }
//      // Directions matter.
//      // Instead of summing over all sources,
//      // return result from each source separately.
//      for (int32 component = 0; component < ncomp; ++component)
//      {
//        length_arr.get_item(size_ray_srcs * index + ray_src, component)
//            = res[component];
//      }
//    }
//
//    delete [] res;
//  });
//  return path_lengths;
}



void UncollidedFlux::execute(DataSet &data_set)
{
  if(m_total_cross_section_field == "")
  {
    DRAY_ERROR("Total cross section field not set");
  }

  if(m_emission_field == "")
  {
    DRAY_ERROR("Emission field not set");
  }

  if(!data_set.has_field(m_total_cross_section_field))
  {
    DRAY_ERROR("No total cross section field '"<<m_total_cross_section_field<<"' found");
  }

  if(!data_set.has_field(m_emission_field))
  {
    DRAY_ERROR("No emission field '"<<m_emission_field<<"' found");
  }

  if(!data_set.has_field(m_overwrite_first_scatter_field))
  {
    DRAY_ERROR("No output first scatter field '"<<m_overwrite_first_scatter_field<<"' found");
  }

  TopologyBase *topo = data_set.topology();
  if(dynamic_cast<UniformTopology*>(topo) != nullptr)
  {
    std::cout<<"Boom\n";

    UniformTopology *uni_topo = dynamic_cast<UniformTopology*>(topo);
    LowOrderField *total_cross_section = dynamic_cast<LowOrderField*>(data_set.field(m_total_cross_section_field));
    LowOrderField *emission = dynamic_cast<LowOrderField*>(data_set.field(m_emission_field));
    LowOrderField *first_scatter_out = dynamic_cast<LowOrderField*>(data_set.field(m_overwrite_first_scatter_field));

    if(total_cross_section->assoc() != LowOrderField::Assoc::Element)
    {
      DRAY_ERROR("Total cross section field must be associated with elements");
    }

    if(emission->assoc() != LowOrderField::Assoc::Element)
    {
      DRAY_ERROR("Emission field must be associated with elements");
    }

    if(first_scatter_out->assoc() != LowOrderField::Assoc::Element)
    {
      DRAY_ERROR("First scatter field must be associated with elements");
    }

    const int32 legendre_order = this->legendre_order();

    const int32 num_moments = (legendre_order+1)*(legendre_order+1);

    if (emission->values().size()
        != total_cross_section->values().size() * num_moments)
    {
      DRAY_ERROR("Emission field must have moments.");
    }

    if (first_scatter_out->values().size()
        != total_cross_section->values().size() * num_moments)
    {
      DRAY_ERROR("First scatter output field must have moments.");
    }

    Array<int32> source_cells;
    Array<Vec<Float,3>> ray_sources = detail::cell_centers_nonzero(*uni_topo, emission, num_moments, source_cells);

    const size_t possible_sources = emission->values().size() / num_moments;
    const size_t actual_sources = source_cells.size();
    /// std::cout << actual_sources << " of " << possible_sources << " cells are sources.\n";

    Array<Vec<Float,3>> destinations = detail::cell_centers(*uni_topo);
    Array<Float> plengths = go_trace(destinations, ray_sources, *uni_topo, total_cross_section);

    const Float cell_volume = (uni_topo->spacing()[0]
                               * uni_topo->spacing()[1]
                               * uni_topo->spacing()[2]);

    Array<Float> destination_moments = detail::integrate_moments(destinations,
                                                                 legendre_order,
                                                                 plengths,
                                                                 ray_sources,
                                                                 source_cells,
                                                                 cell_volume,
                                                                 emission);

    detail::copy_moments(destination_moments, num_moments, first_scatter_out);
    std::cout << "Uncollided flux.\n";
  }
  else
  {
    DRAY_ERROR("UncollidedFlux filter only supports UniformTopology");
  }
}

void
UncollidedFlux::domain_data(Collection &collection)
{
  std::stringstream msg;
  bool valid = true;
  // these are safe to check in mpi
  if(m_total_cross_section_field == "")
  {
    DRAY_ERROR("Total cross section field not set");
  }

  if(m_emission_field == "")
  {
    DRAY_ERROR("Emission field not set");
  }

  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet data_set = collection.domain(i);

    if(!data_set.has_field(m_total_cross_section_field))
    {
      valid = false;
      msg<<"No total cross section field '"<<m_total_cross_section_field<<"' found";
    }

    if(!data_set.has_field(m_emission_field))
    {
      valid = false;
      msg<<"No emission field '"<<m_emission_field<<"' found";
    }

    if(!data_set.has_field(m_overwrite_first_scatter_field))
    {
      valid = false;
      msg<<"No output first scatter field '"<<m_overwrite_first_scatter_field<<"' found";
    }

    if(valid)
    {
      TopologyBase *topo = data_set.topology();
      if(dynamic_cast<UniformTopology*>(topo) != nullptr)
      {
        DomainData data;
        data.m_topo = dynamic_cast<UniformTopology*>(topo);
        data.m_cross_section = dynamic_cast<LowOrderField*>(data_set.field(m_total_cross_section_field));
        m_num_groups = data.m_cross_section->values().ncomp();
        if(data.m_cross_section== nullptr)
        {
          valid = false;
          msg<<"Bad cross section\n";
        }
        data.m_source = dynamic_cast<LowOrderField*>(data_set.field(m_emission_field));
        if(data.m_source == nullptr)
        {
          valid = false;
          msg<<"Bad source\n";
        }
        m_domain_data.push_back(data);
      }
      else
      {
        valid = false;
        msg<<"Not a uniform domain\n";
      }
    }
  }

  if(!valid)
  {
    DRAY_ERROR("Bad things "<<msg.str());
  }

}

void
UncollidedFlux::recreate_rays(Array<Vec<Float,3>> sources,
                              Array<int32> &pixel_ids,
                              Array<Float> &path_lengths,
                              std::vector<Array<Ray>> &domain_rays,
                              std::vector<Array<Float>> &domain_path_lengths)
{
  // we created the rays with a specific indexing
  // pixel_id  = cell * num_source + source_idx
  // the cell_id here is a global cell id
  // One might ask why don't you just send all the rays over MPI?
  // 1) thats a lot of extra duplicate data
  // 2) Its complicated to send structs over mpi and simple to rebuild the rays

  // count the number of cells and create counts
  // for each domain for global indexing
  int32 total_cells = 0;
  Array<int32> domain_offsets;
  domain_offsets.resize(m_global_coords.size());
  int32 *h_offsets_ptr = domain_offsets.get_host_ptr();
  // keep track of the first domain on this rank so we get the local domain
  int32 local_start = -1;
  for(int32 i = 0; i < m_global_coords.size(); ++i)
  {
    UniformData &c = m_global_coords[i];

    if(local_start == -1 && c.m_rank == dray::mpi_rank())
    {
      local_start = i;
    }

    int32 cells = c.m_dims[0] * c.m_dims[1] * c.m_dims[2];
    total_cells += cells;

    if(i == 0)
    {
      h_offsets_ptr[0] = cells;
    }
    else
    {
      h_offsets_ptr[i] = h_offsets_ptr[i - 1] + cells;
    }
  }

  std::cout<<"Local start "<<local_start<<"\n";
  const int32 out_size = pixel_ids.size();

  Array<Ray> rays;
  rays.resize(out_size);
  // we will need to gather rays to the local domains,
  // so we will keep track of the domain as we recreate the ray
  Array<int32> local_domains;
  local_domains.resize(out_size);
  int32 *local_domain_ptr = local_domains.get_device_ptr();

  Array<Vec<Float,3>> origins;
  Array<Vec<Float,3>> spacings;
  Array<Vec<int32,3>> dims;
  detail::pack_coords(m_global_coords, origins, spacings, dims);

  const int32 source_size = sources.size();
  const int32 num_domains = m_global_coords.size();

  Ray *ray_ptr = rays.get_device_ptr();
  const Vec<Float,3> *source_ptr = sources.get_device_ptr_const();
  const Vec<Float,3> *origin_ptr = origins.get_device_ptr_const();
  const Vec<Float,3> *space_ptr = spacings.get_device_ptr_const();
  const Vec<int32,3> *dims_ptr = dims.get_device_ptr_const();
  const int32 *domain_offsets_ptr = domain_offsets.get_device_ptr_const();

  const int32 *pixel_id_ptr = pixel_ids.get_device_ptr();

  // if we went big, we can't be using int32s here for cell_id
  RAJA::forall<for_policy> (RAJA::RangeSegment(0, out_size), [=] DRAY_LAMBDA (int32 ii)
  {
    int32 domain_id = 0;
    const int32 pixel_id = pixel_id_ptr[ii];
    const int32 cell = pixel_id / source_size;
    const int32 source_id = pixel_id % source_size;

    while(cell >= domain_offsets_ptr[domain_id])
    {
      domain_id++;
    }


    local_domain_ptr[ii] = domain_id - local_start;

    detail::UniformLocator locator(origin_ptr[domain_id],
                                   space_ptr[domain_id],
                                   dims_ptr[domain_id]);

    int32 local_cell = domain_id == 0 ? cell : cell - domain_offsets_ptr[domain_id - 1];

    Vec<Float,3> center = locator.cell_center(local_cell);

    Ray ray;
    Vec<Float,3> source = source_ptr[source_id];
    Vec<Float,3> dir = center - source;

    ray.m_far = dir.magnitude();
    dir.normalize();
    ray.m_dir = dir;
    ray.m_orig = source;
    ray.m_near = 0.f;
    // re-id the rays to the local cells
    ray.m_pixel_id = local_cell;
    // todo add ray data with destination == rank
    ray_ptr[ii] = ray;
  });

  const int32 local_size = m_domain_data.size();
  domain_rays.resize(local_size);
  domain_path_lengths.resize(local_size);

  for(int32 domain = 0; domain < local_size; ++domain)
  {

    Array<int32> gather_flags;
    gather_flags.resize(out_size);
    int32 *flags_ptr = gather_flags.get_device_ptr();

    RAJA::forall<for_policy> (RAJA::RangeSegment(0, out_size), [=] DRAY_LAMBDA (int32 ii)
    {
      int32 flag = local_domain_ptr[ii] == domain ? 1 : 0;
      flags_ptr[ii] = flag;
    });

    Array<int32> sparse_ids = index_flags(gather_flags);
    const int32 size = sparse_ids.size();
    domain_rays[domain].resize(size);

    const int32 num_groups = m_num_groups;
    domain_path_lengths[domain].resize(size,num_groups);
    Ray *local_ray_ptr = domain_rays[domain].get_device_ptr();
    Float *local_data_ptr = domain_path_lengths[domain].get_device_ptr();

    const int32 *sparse_ids_ptr = sparse_ids.get_device_ptr();
    const Float *data_ptr = path_lengths.get_device_ptr_const();

    RAJA::forall<for_policy> (RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 ii)
    {
      const int32 sparse_idx = sparse_ids_ptr[ii];
      local_ray_ptr[ii] = ray_ptr[sparse_idx];
      const int32 sparse_offset = sparse_idx * num_groups;
      const int32 offset = ii * num_groups;
      for(int32 group = 0; group < num_groups; ++group)
      {
        local_data_ptr[offset + group] = data_ptr[sparse_offset + group];
      }
    });

  }

}

std::vector<Array<Ray>>
UncollidedFlux::create_rays(Array<Vec<Float,3>> sources)
{
  // count the number of cells and create counts
  // for each domain for indexing
  int32 total_cells = 0;
  Array<int32> domain_offsets;
  domain_offsets.resize(m_global_coords.size());
  int32 *h_offsets_ptr = domain_offsets.get_host_ptr();
  for(int32 i = 0; i < m_global_coords.size(); ++i)
  {
    UniformData &c = m_global_coords[i];
    int32 cells = c.m_dims[0] * c.m_dims[1] * c.m_dims[2];
    std::cout<<"Domain "<<i<<" dims "<<c.m_dims<<" cells "<<cells<<"\n";
    total_cells += cells;

    if(i == 0)
    {
      h_offsets_ptr[0] = cells;
    }
    else
    {
      h_offsets_ptr[i] = h_offsets_ptr[i - 1] + cells;
    }
    std::cout<<"offset "<<h_offsets_ptr[i]<<"\n";

  }

  int32 total_rays = total_cells * sources.size();
  // this could be a really big number
  if(total_rays < 0)
  {
    // overflow.
    // Arrays use size_t, so we are probably ok.
    DRAY_ERROR("I knew this day would come.");
  }

  if(dray::mpi_rank() == 0)
  {
    std::cout<<"total cells "<<total_cells<<"\n";
    std::cout<<"total sources "<<sources.size()<<"\n";
    std::cout<<"total rays "<<total_cells * sources.size()<<"\n";
    std::cout<<"Num groups "<<m_num_groups<<"\n";
  }

  Array<Ray> rays;
  rays.resize(total_rays);
  Array<Vec<Float,3>> origins;
  Array<Vec<Float,3>> spacings;
  Array<Vec<int32,3>> dims;
  detail::pack_coords(m_global_coords, origins, spacings, dims);

  const int32 source_size = sources.size();
  const int32 num_domains = m_global_coords.size();

  Ray *ray_ptr = rays.get_device_ptr();
  const Vec<Float,3> *source_ptr = sources.get_device_ptr_const();
  const Vec<Float,3> *origin_ptr = origins.get_device_ptr_const();
  const Vec<Float,3> *space_ptr = spacings.get_device_ptr_const();
  const Vec<int32,3> *dims_ptr = dims.get_device_ptr_const();
  const int32 *domain_offsets_ptr = domain_offsets.get_device_ptr_const();

  // if we went big, we can't be using int32s here for cell_id
  RAJA::forall<for_policy> (RAJA::RangeSegment(0, total_cells), [=] DRAY_LAMBDA (int32 cell)
  {
    int32 domain_id = 0;
    while(cell >= domain_offsets_ptr[domain_id])
    {
      domain_id++;
    }

    detail::UniformLocator locator(origin_ptr[domain_id],
                                   space_ptr[domain_id],
                                   dims_ptr[domain_id]);

    int32 local_cell = domain_id == 0 ? cell : cell - domain_offsets_ptr[domain_id - 1];

    Vec<Float,3> center = locator.cell_center(local_cell);

    for(int32 i = 0; i < source_size; ++i)
    {
      Ray ray;
      Vec<Float,3> source = source_ptr[i];
      Vec<Float,3> dir = center - source;

      ray.m_far = dir.magnitude();
      dir.normalize();
      ray.m_dir = dir;
      ray.m_orig = source;
      ray.m_near = 0.f;
      ray.m_pixel_id = cell * source_size + i;
      // todo add ray data with destination == rank
      ray_ptr[cell * source_size + i] = ray;
    }
  });

  std::vector<UniformData> local_data;
  detail::gather_local_uniform_data(m_domain_data, local_data);
  Array<Vec<Float,3>> local_origins;
  Array<Vec<Float,3>> local_spacings;
  Array<Vec<int32,3>> local_dims;
  detail::pack_coords(local_data, local_origins, local_spacings, local_dims);

  Array<int32> keep;
  keep.resize(total_rays);
  int32 *keep_ptr = keep.get_device_ptr();

  const int32 local_size = local_data.size();
  const Vec<Float,3> *l_origin_ptr = local_origins.get_device_ptr_const();
  const Vec<Float,3> *l_space_ptr = local_spacings.get_device_ptr_const();
  const Vec<int32,3> *l_dims_ptr = local_dims.get_device_ptr_const();

  // ok now lets get rid of everything we don't need
  RAJA::forall<for_policy> (RAJA::RangeSegment(0, total_rays), [=] DRAY_LAMBDA (int32 ii)
  {
    Ray ray = ray_ptr[ii];
    int32 keep = 0;

    for(int32 i = 0; i < local_size; ++i)
    {
      detail::UniformLocator locator(l_origin_ptr[i],
                                     l_space_ptr[i],
                                     l_dims_ptr[i]);

      if(locator.hit(ray) != infinity<Float>())
      {
        keep = 1;
      }

      // for rays cast to and from the same cell:
      // technically, the far should be 0 so hit will fail
      // but it makes me nervous so just double check here
      // TODO: just make this some eps based on spacing
      if(ray.m_far < 0.001)
      {
        keep = 0;
      }
    }
    keep_ptr[ii] = keep;



  });

  Array<int32> compact_idxs = index_flags(keep);
  rays = gather (rays, compact_idxs);

  std::cout<<"keeping "<<rays.size()<<" rays\n";

  // we have to re-grab the ray pointer
  ray_ptr = rays.get_device_ptr();
  const int32 rank_ray_size = rays.size();
  std::vector<Array<Ray>> domain_rays;
  domain_rays.resize(local_size);

  Array<int32> dom_flags;
  dom_flags.resize(rank_ray_size);
  int32 *dom_flags_ptr = dom_flags.get_device_ptr();

  for(int32 i = 0; i < local_size; ++i)
  {
    const int32 local_domain = i;
    // count the remaining rays and figure out which domains get them
    RAJA::forall<for_policy> (RAJA::RangeSegment(0, rank_ray_size), [=] DRAY_LAMBDA (int32 ii)
    {
      Ray ray = ray_ptr[ii];
      int32 keep = 0;

      detail::UniformLocator locator(l_origin_ptr[local_domain],
                                     l_space_ptr[local_domain],
                                     l_dims_ptr[local_domain]);
      if(locator.hit(ray) != infinity<Float>())
      {
        keep = 1;
      }

      dom_flags_ptr[ii] = keep;
    });

    Array<int32> compact_idxs = index_flags(dom_flags);
    domain_rays[i] = gather (rays, compact_idxs);
    std::cout<<"domain "<<i<<" rays "<<domain_rays[i].size()<<"\n";
  }
  return domain_rays;
}

std::vector<Array<Float>>
UncollidedFlux::init_data(std::vector<Array<Ray>> &rays)
{
  const int32 local_size = rays.size();

  // initialize energy groups
  std::vector<Array<Float>> ray_data;
  ray_data.resize(local_size);

  std::vector<UniformData> local_data;
  detail::gather_local_uniform_data(m_domain_data, local_data);
  Array<Vec<Float,3>> local_origins, local_spacings;
  Array<Vec<int32,3>> local_dims;
  detail::pack_coords(local_data, local_origins, local_spacings, local_dims);

  // TODO: make a container for this?
  const Vec<Float,3> *l_origin_ptr = local_origins.get_device_ptr_const();
  const Vec<Float,3> *l_space_ptr = local_spacings.get_device_ptr_const();
  const Vec<int32,3> *l_dims_ptr = local_dims.get_device_ptr_const();

  for(int32 i = 0; i < local_size; ++i)
  {
    const int32 local_domain = i;
    const int32 local_rays = rays[i].size();
    ray_data[i].resize(local_rays, m_num_groups);
    const Ray *local_rays_ptr = rays[i].get_device_ptr_const();
    NonConstDeviceArray<Float> local_data(ray_data[i]);
    const int32 num_groups = m_num_groups;

    // we have already checked that this exists
    ConstDeviceArray<Float> emission_dev(m_domain_data[i].m_source->values());
    const int32 legendre_order = this->legendre_order();
    const int32 num_moments = (legendre_order + 1) * (legendre_order + 1);

    RAJA::forall<for_policy> (RAJA::RangeSegment(0, local_rays), [=] DRAY_LAMBDA (int32 ii)
    {
      Ray ray = local_rays_ptr[ii];

      for(int32 i = 0; i < num_groups; ++i)
      {
        local_data.get_item(ii,i) = 1.f;
      }

      detail::UniformLocator locator(l_origin_ptr[local_domain],
                                     l_space_ptr[local_domain],
                                     l_dims_ptr[local_domain]);

      bool inside = locator.is_inside(ray.m_orig);
      // the only way the origin of the ray is inside is if
      // if originates from a source cell
      if(inside)
      {
        // begin the stuff i don't know about
        int32 source_cell = locator.locate(ray.m_orig);
        SphericalHarmonics<Float> sph(legendre_order);
        const Float * sph_eval = sph.eval_all(ray.m_dir);

        for (int32 group = 0; group < num_groups; ++group)
        {
          // Evaluate emission in the direction of omega_hat.
          Float dEmission_dV = 0.0f;
          for (int32 nm = 0; nm < num_moments; ++nm)
          {
            dEmission_dV += sph_eval[nm]
                           * emission_dev.get_item(num_moments * source_cell + nm,
                                                   group);
          }
          local_data.get_item(ii,group) = dEmission_dV;
        }
      }

    });
  }
  return ray_data;
}

void ///Collection
UncollidedFlux::execute(Collection &collection)
{
  domain_data(collection);
  detail::gather_uniform_data(m_domain_data, m_global_coords);

  std::vector<Array<int32>> source_cells;
  std::vector<Array<Vec<Float,3>>> ray_sources;
  source_cells.resize(m_domain_data.size());
  ray_sources.resize(m_domain_data.size());

  const int32 num_moments = (m_legendre_order + 1) * (m_legendre_order + 1);

  for(int32 i = 0; i < m_domain_data.size(); ++i)
  {
    DomainData data = m_domain_data[i];

    ray_sources[i] = detail::cell_centers_nonzero(*data.m_topo,
                                                  data.m_source,
                                                  num_moments,
                                                  source_cells[i]);
    if(dray::mpi_rank() == 0)
    {
      std::cout<<"Ray sources "<<i<<" "<<ray_sources[i].size()<<"\n";
    }
  }

  // put all the sources into a single array
  Array<Vec<Float,3>> sources = detail::gather_sources(ray_sources);
  std::vector<Array<Ray>> rays = create_rays(sources);
  std::vector<Array<Float>> ray_data = init_data(rays);

  // now go bananas and do the tracing
  go_bananas(rays, ray_data);

  // exchange takes in rays for each domain, figures out what rank they belong
  // to and does a global exchange to send the data to the owning ranks
  // further, all rays are composited into one batch per rank.
  Array<Float> path_lengths;
  Array<int32> pixel_ids;
  exchange(rays, ray_data, pixel_ids, path_lengths);

  // we now have all the ray data that belongs to this rank,
  // Now, we have to figure out which domains they belong to.
  // During this recreation, we now change the pixel_id from
  // a global ordering to the id of the destination cell
  // Further, they have been sorted in by global id, which means
  // that the rays are actually in order
  std::vector<Array<Ray>> domain_rays;
  std::vector<Array<Float>> domain_path_lengths;
  recreate_rays(sources,pixel_ids, path_lengths, domain_rays, domain_path_lengths);

  for(int32 i = 0; i < domain_rays.size(); ++i)
  {

    const int32 legendre_order = this->legendre_order();
    int32 num_cells = m_domain_data[i].m_topo->cells();
    Float cell_volume = m_domain_data[i].m_topo->spacing()[0] *
                        m_domain_data[i].m_topo->spacing()[1] *
                        m_domain_data[i].m_topo->spacing()[2];

    Array<Float> moments = detail::integrate_moments(domain_rays[i],
                                                     domain_path_lengths[i],
                                                     legendre_order,
                                                     cell_volume,
                                                     num_cells);
  }
}


void UncollidedFlux::total_cross_section_field(const std::string field_name)
{
  m_total_cross_section_field = field_name;
}

void UncollidedFlux::emission_field(const std::string field_name)
{
  m_emission_field = field_name;
}

void UncollidedFlux::overwrite_first_scatter_field(const std::string field_name)
{
  m_overwrite_first_scatter_field = field_name;
}

int32 UncollidedFlux::legendre_order() const
{
  return m_legendre_order;
}

void UncollidedFlux::legendre_order(int32 l_order)
{
  m_legendre_order = l_order;
}

void UncollidedFlux::uniform_isotropic_scattering(Float sigs)
{
  m_sigs = sigs;
}

inline
int32 moment_to_legendre(int32 nm)
{
  return int32(sqrt(nm));
}

};//namespace dray

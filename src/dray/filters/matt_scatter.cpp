
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

#ifdef DRAY_MPI_ENABLED
#include <mpi.h>
#endif



namespace dray
{

namespace detail
{

struct UniformData
{
  Vec<Float,3> m_spacing;
  Vec<Float,3> m_origin;
  Vec<int32,3> m_dims;
  int32 m_rank;
};

// CPU only for comm
struct RayData
{
  int64 m_ray_id;
  Float m_seg_length;
  Float *m_group_optical_depths;
};

#ifdef DRAY_MPI_ENABLED
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
  uniform_data.resize(local_size);
  for(int32 i = 0; i < local_size; ++i)
  {
    uniform_data[i].m_spacing = dom_data[i].m_topo->spacing();
    uniform_data[i].m_origin = dom_data[i].m_topo->origin();
    uniform_data[i].m_dims = dom_data[i].m_topo->cell_dims();
    uniform_data[i].m_rank = 0;
  }
#endif
}
Array<Vec<Float,3>> gather_sources(std::vector<Array<Vec<Float,3>>> &source_list)
{
  int32 list_size = source_list.size();
  if(list_size == 0)
  {
    DRAY_ERROR("Empty source list");
  }

  if(list_size == 1)
  {
    return source_list[0];
  }

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
//#ifdef DRAY_MPI_ENABLED
//  int32 rank, procs;
//
//  MPI_Comm comm = MPI_Comm_f2c(dray::mpi_comm());
//  MPI_Comm_rank(comm, &rank);
//  MPI_Comm_size(comm, &procs);
//
//  std::vector<int32> dom_counts;
//  dom_counts.resize(procs);
//  MPI_Allgather(&local_size, 1, MPI_INT, &dom_counts[0], 1, MPI_INT, comm);
//
//  int32 global_size = 0;
//  for(auto count : dom_counts)
//  {
//    global_size += count;
//  }
//  std::cout<<"Global size "<<global_size<<"\n";
//
//  // setup the send and recv buffers
//  std::vector<Float> global_float_data;
//  global_float_data.resize(global_size * 6);
//
//  std::vector<Float> local_float_data;
//  local_float_data.resize(local_size * 6);
//
//  std::vector<int32> global_int_data;
//  global_int_data.resize(global_size * 3);
//
//  std::vector<int32> local_int_data;
//  local_int_data.resize(local_size * 3);
//
//  for(int32 i = 0; i < local_size; ++i)
//  {
//    Vec<Float,3> spacing = dom_data[i].m_topo->spacing();
//    Vec<Float,3> origin = dom_data[i].m_topo->origin();
//    Vec<int32,3> dims = dom_data[i].m_topo->cell_dims();
//
//    local_float_data[i * 6 + 0] = spacing[0];
//    local_float_data[i * 6 + 1] = spacing[1];
//    local_float_data[i * 6 + 2] = spacing[2];
//    local_float_data[i * 6 + 3] = origin[0];
//    local_float_data[i * 6 + 4] = origin[1];
//    local_float_data[i * 6 + 5] = origin[2];
//
//    local_int_data[i * 3 + 0] = dims[0];
//    local_int_data[i * 3 + 1] = dims[1];
//    local_int_data[i * 3 + 2] = dims[2];
//  }
//
//  std::vector<int32> recv_counts;
//  recv_counts.resize(procs);
//  for(int32 i = 0; i < procs; ++i)
//  {
//    recv_counts[i] = dom_counts[i] * 6;
//  }
//  allgatherv(local_float_data, global_float_data, recv_counts, comm);
//
//  for(int32 i = 0; i < procs; ++i)
//  {
//    recv_counts[i] = dom_counts[i] * 3;
//  }
//  allgatherv(local_int_data, global_int_data, recv_counts, comm);
//
//  uniform_data.resize(global_size);
//  int32 curr_rank = 0;
//  int32 counter = dom_counts[0];
//  for(int32 i = 0; i < global_size; ++i)
//  {
//    uniform_data[i].m_spacing[0] = global_float_data[i * 6 + 0];
//    uniform_data[i].m_spacing[1] = global_float_data[i * 6 + 1];
//    uniform_data[i].m_spacing[2] = global_float_data[i * 6 + 2];
//    uniform_data[i].m_origin[0] = global_float_data[i * 6 + 3];
//    uniform_data[i].m_origin[1] = global_float_data[i * 6 + 4];
//    uniform_data[i].m_origin[2] = global_float_data[i * 6 + 5];
//
//    uniform_data[i].m_dims[0] = global_int_data[i * 3 + 0];
//    uniform_data[i].m_dims[1] = global_int_data[i * 3 + 1];
//    uniform_data[i].m_dims[2] = global_int_data[i * 3 + 2];
//
//    if( i >= counter)
//    {
//      curr_rank++;
//      counter += dom_counts[curr_rank];
//    }
//    uniform_data[i].m_rank = curr_rank;
//
//  }
//
//#else
//  uniform_data.resize(local_size);
//  for(int32 i = 0; i < local_size; ++i)
//  {
//    uniform_data[i].m_spacing = dom_data[i].m_topo->spacing();
//    uniform_data[i].m_origin = dom_data[i].m_topo->origin();
//    uniform_data[i].m_dims = dom_data[i].m_topo->cell_dims();
//    uniform_data[i].m_rank = 0;
//  }
//#endif
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

void ///Collection
UncollidedFlux::execute(Collection &collection)
{
  domain_data(collection);
  std::vector<detail::UniformData> uni_data;
  detail::gather_uniform_data(m_domain_data, uni_data);

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

  Array<Vec<Float,3>> sources = detail::gather_sources(ray_sources);

  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet data_set = collection.domain(i);
    if(data_set.topology()->dims() == 3)
    {
      /// DataSet result_data_set =
      this->execute(data_set);
      /// res.add_domain(result_data_set);
    }
    else
    {
      // just pass it through
      /// res.add_domain(data_set);
    }
  }
  /// return res;
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


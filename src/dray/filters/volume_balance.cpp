#include <dray/filters/volume_balance.hpp>

#include <dray/dray.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/error_check.hpp>
#include <numeric>
#include <algorithm>

#include <dray/filters/subset.hpp>
#include <dray/filters/redistribute.hpp>

#include <dray/GridFunction/device_mesh.hpp>
#include <dray/dispatcher.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>

#ifdef DRAY_MPI_ENABLED
#include<mpi.h>
#endif

namespace dray
{

namespace detail
{

template<typename MeshElement>
void mask_cells(Mesh<MeshElement> &mesh,
                int32 comp,
                float32 min_coord,
                float32 max_coord,
                Array<int32> &mask)
{
  DRAY_LOG_OPEN("mask_cells");

  const int32 num_elems = mesh.get_num_elem();
  DeviceMesh<MeshElement> device_mesh(mesh);

  mask.resize(num_elems);
  int32 *mask_ptr = mask.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 i)
  {
    int32 mask_val = 0;
    MeshElement element = device_mesh.get_elem(i);
    AABB<3> elem_bounds;
    element.get_bounds(elem_bounds);
    float32 center = elem_bounds.m_ranges[comp].center();
    if(center >= min_coord && center < max_coord)
    {
      mask_val = 1;
    }

    mask_ptr[i] = mask_val;
  });


  DRAY_LOG_CLOSE();
}

template<typename MeshElement>
void aabb_cells(Mesh<MeshElement> &mesh,
                Array<AABB<3>> &aabbs)
{
  DRAY_LOG_OPEN("aabb_cells");

  const int32 num_elems = mesh.get_num_elem();
  DeviceMesh<MeshElement> device_mesh(mesh);

  aabbs.resize(num_elems);
  AABB<3> *aabb_ptr = aabbs.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elems), [=] DRAY_LAMBDA (int32 i)
  {
    int32 mask_val = 0;
    MeshElement element = device_mesh.get_elem(i);
    AABB<3> elem_bounds;
    element.get_bounds(elem_bounds);
    aabb_ptr[i] = elem_bounds;
  });

  DRAY_LOG_CLOSE();
}

struct MaskFunctor
{
  Array<int32> m_mask;
  int32 m_dim;
  float32 m_min;
  float32 m_max;
  MaskFunctor(const int32 dim,
              const float32 min,
              const float32 max)
    : m_dim(dim),
      m_min(min),
      m_max(max)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    mask_cells(topo.mesh(), m_dim, m_min, m_max, m_mask);
  }
};

struct AABBFunctor
{
  Array<AABB<3>> m_aabbs;
  AABBFunctor()
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    aabb_cells(topo.mesh(), m_aabbs);
  }
};


void split(const DomainTask &task,
           DataSet &dataset,
           Collection &col,
           std::vector<int32> &dests)
{
  DRAY_LOG_OPEN("split");

  AABB<3> bounds = dataset.topology()->bounds();
  const int32 max_comp = bounds.max_dim();
  float32 length = bounds.m_ranges[max_comp].length();
  const float32 volume = bounds.volume();
  DRAY_LOG_ENTRY("box_length", length);
  DRAY_LOG_ENTRY("box_volume", bounds.volume());
  DRAY_LOG_ENTRY("box_min", bounds.m_ranges[max_comp].min());

  float32 total = task.m_amount;

  std::vector<float32> pieces;
  pieces.resize(task.m_splits.size() + 1);
  pieces[0] = task.m_amount;
  DRAY_LOG_ENTRY("split_0", task.m_amount);

  for(int32 i = 0; i < task.m_splits.size(); ++i)
  {
    pieces[i+1] = task.m_splits[i].first;
    total += task.m_splits[i].first;
    DRAY_LOG_ENTRY("split", task.m_splits[i].first);
  }

  const int32 num_pieces = pieces.size();
  std::vector<float32> normalized_volume;
  normalized_volume.resize(num_pieces);

  // normalize
  for(int32 i = 0; i < num_pieces; ++i)
  {
    pieces[i] /= total; 
    normalized_volume[i] = pieces[i];
    DRAY_LOG_ENTRY("normalized_length", pieces[i]);
  }

  // scale 
  for(int32 i = 0; i < num_pieces; ++i)
  {
    pieces[i] *= length; 
    DRAY_LOG_ENTRY("piece_length", pieces[i]);
  }

  AABBFunctor aabb_func;
  dispatch(dataset.topology(), aabb_func);
  Array<AABB<3>> aabbs = aabb_func.m_aabbs;
  AABB<3> *aabbs_ptr = aabbs.get_host_ptr();

  stable_sort(aabbs_ptr, aabbs_ptr + aabbs.size(),
       [&max_comp](const AABB<3> &i1, const AABB<3> &i2)
       {
         return i1.center()[max_comp] < i2.center()[max_comp];
       });

  float32 aabb_tot_vol = 0;
  for(int i = 0; i < aabbs.size(); ++i)
  {
    aabb_tot_vol += aabbs_ptr[i].volume();
  }
  DRAY_LOG_ENTRY("aabbs_vol", aabb_tot_vol);

  std::vector<int32> divs;
  divs.resize(num_pieces);
  float32 curr_vol = 0;
  float32 piece_idx = 0;

  for(int32 i = 0; i < aabbs.size() && piece_idx < num_pieces; ++i)
  {
    curr_vol += aabbs_ptr[i].volume() / aabb_tot_vol; 
    if(curr_vol >= normalized_volume[piece_idx])
    {
      divs[piece_idx] = i;
      DRAY_LOG_ENTRY("div_normal", curr_vol);
      piece_idx++;
      curr_vol = 0;
    }
  }
  divs[num_pieces - 1] = aabbs.size() - 1;

  for(int32 i = 0; i < num_pieces; ++i)
  {
    DRAY_LOG_ENTRY("div", divs[i]);;
  }


  std::vector<float32> ranges;
  ranges.resize(num_pieces+1);
  ranges[0] = bounds.m_ranges[max_comp].min();
  // bump it by an epsilon
  ranges[num_pieces] = bounds.m_ranges[max_comp].max() + length * 1e-3;

  //for(int32 i = 0; i < num_pieces - 1; ++i)
  //{
  //  ranges[i+1] = ranges[i] + pieces[i];
  //}

  for(int32 i = 0; i < num_pieces-1; ++i)
  {
    int32 idx = divs[i];
    ranges[i+1] = aabbs_ptr[idx].center()[max_comp] + length * 1e-3;
  }


  for(int32 i = 0; i < num_pieces+1; ++i)
  {
    DRAY_LOG_ENTRY("range", ranges[i]);
  }

  for(int32 i = 0; i < ranges.size() - 1; ++i)
  {
    MaskFunctor func(max_comp,ranges[i],ranges[i+1]);
    dispatch(dataset.topology(), func);
    Subset subset;
    DataSet piece = subset.execute(dataset, func.m_mask);
    DRAY_LOG_ENTRY("piece_length",ranges[i+1] - ranges[i]);
    DRAY_LOG_ENTRY("target",normalized_volume[i]);
    DRAY_LOG_ENTRY("efficiency",normalized_volume[i] / (piece.topology()->bounds().volume() / volume));

    if(piece.topology()->cells() > 0)
    {
      col.add_domain(piece);
      if(i == 0)
      {
        dests.push_back(dray::mpi_rank());
      }
      else
      {
        dests.push_back(task.m_splits[i-1].second);
      }
    }
  }
  DRAY_LOG_CLOSE();
}


}//namespace detail

VolumeBalance::VolumeBalance()
{
}

float32
VolumeBalance::perfect_splitting(std::vector<RankTasks> &distribution)
{
  const int32 size = distribution.size();

  std::vector<int32> idx(size);
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
       [&distribution](int32 i1, int32 i2)
       {
         return distribution[i1].amount() < distribution[i2].amount();
       });

  float32 sum = 0;
  for(int32 i = 0; i < size; ++i)
  {
    sum += distribution[i].amount();
  }
  const float32 ave = sum / float32(size);

  int32 giver = size - 1;
  int32 taker = 0;
  float32 eps = distribution[idx[giver]].amount() * 1e-3;
  float32 max_val = distribution[idx[giver]].amount();
  //std::cout<<"Taker "<<taker<<"\n";
  while(giver > taker)
  {
    //std::cout<<"Giver "<<giver<<" taker "<<taker<<"\n";
    int32 giver_idx = idx[giver];
    int32 taker_idx = idx[taker];
    float32 giver_work = distribution[giver_idx].amount();
    if(giver_work < ave)
    {
      break;
    }

    float32 giver_dist = giver_work - ave;

    float32 taker_dist = ave - distribution[taker_idx].amount();
    float32 give_amount = 0;
    if(taker_dist > giver_dist)
    {
      give_amount = giver_dist;
    }
    else
    {
      give_amount = taker_dist;
    }

    give_amount = distribution[giver_idx].split_biggest(give_amount,taker_idx);
    distribution[taker_idx].add(give_amount);

    // does giver have more?
    if(distribution[giver_idx].amount() <= ave + eps)
    {
      giver--;
    }
    // can taker take more?
    if(distribution[taker_idx].amount() >= ave - eps)
    {
      taker++;
    }
  }
  //std::cout<<"giver "<<giver<<" taker "<<taker<<"\n";

  float32 max_after = 0;
  for(int32 i = 0; i < size; ++i)
  {
    max_after = std::max(distribution[i].amount(), max_after);
  }

  if(dray::mpi_rank() == 0)
  {
    std::ofstream load_file;
    load_file.open ("perfect_splits.txt");
    for(auto &l : distribution)
    {
      load_file << l.amount() <<"\n";
    }
    load_file.close();
  }
  return max_after / max_val;
}

Collection
VolumeBalance::execute(Collection &collection, Camera &camera)
{
  DRAY_LOG_OPEN("volume_balance");
  Collection res;

  const int32 local_doms = collection.local_size();
  std::vector<float32> local_volumes;
  local_volumes.resize(local_doms);

  float32 total_volume = 0;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dataset = collection.domain(i);
    AABB<3> bounds = dataset.topology()->bounds();
    float32 pixels = static_cast<float32>(camera.subset_size(bounds));
    local_volumes[i] = bounds.volume() * pixels;
    total_volume += local_volumes[i];
  }

  DRAY_LOG_ENTRY("local_volume", total_volume);


#ifdef DRAY_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  const int32 comm_size = dray::mpi_size();
  const int32 rank = dray::mpi_rank();
  const int32 global_size = collection.size();

  std::vector<int32> global_counts;
  global_counts.resize(comm_size);
  MPI_Allgather(&local_doms, 1, MPI_INT, &global_counts[0], 1, MPI_INT, mpi_comm);

  std::vector<int32> global_offsets;
  global_offsets.resize(comm_size);
  global_offsets[0] = 0;

  for(int i = 1; i < comm_size; ++i)
  {
    global_offsets[i] = global_offsets[i-1] + global_counts[i-1];
  }

  std::vector<float32> global_volumes;
  global_volumes.resize(global_size);

  std::vector<float32> rank_volumes;
  rank_volumes.resize(comm_size);

  MPI_Allgather(&total_volume,1, MPI_FLOAT,&rank_volumes[0],1,MPI_FLOAT,mpi_comm);

  MPI_Allgatherv(&local_volumes[0],
                 local_doms,
                 MPI_FLOAT,
                 &global_volumes[0],
                 &global_counts[0],
                 &global_offsets[0],
                 MPI_FLOAT,
                 mpi_comm);


  std::vector<RankTasks> distribution;
  distribution.resize(comm_size);
  for(int32 i = 0; i < comm_size; ++i)
  {
    distribution[i].m_rank = i;
    for(int32 t = 0; t < global_counts[i]; ++t)
    {
      const int32 offset = global_offsets[i];
      distribution[i].add_task(global_volumes[offset + t]);
    }
  }
  float32 ratio = perfect_splitting(distribution);
  if(rank == 0)
  {
    std::cout<<"Ratio "<<ratio<<"\n";
  }

  std::vector<int32> src_list;
  std::vector<int32> dest_list;
  Collection chopped = chopper(distribution, collection, src_list, dest_list);
  DRAY_LOG_ENTRY("chopped_local_domains", chopped.local_size());
  //if(rank == 0)
  //{
  //  for(int i = 0; i < src_list.size(); ++i)
  //  {
  //    std::cout<<"src "<<src_list[i]<<" -> "<<dest_list[i]<<"\n";
  //  }
  //}

  //map(distribution, src_list, dest_list);

  Redistribute redist;
  res = redist.execute(chopped, src_list, dest_list);
  DRAY_LOG_ENTRY("result_local_domains", res.local_size());
#endif

  total_volume = 0;
  for(int32 i = 0; i < res.local_size(); ++i)
  {
    DataSet dataset = res.domain(i);
    AABB<3> bounds = dataset.topology()->bounds();
    float32 pixels = static_cast<float32>(camera.subset_size(bounds));
    total_volume += bounds.volume() * pixels;
  }
  DRAY_LOG_ENTRY("result_local_volume", total_volume);
  DRAY_LOG_CLOSE();
  return res;
}

Collection
VolumeBalance::chopper(const std::vector<RankTasks> &distribution,
                       Collection &collection,
                       std::vector<int32> &src_list,
                       std::vector<int32> &dest_list)
{
  Collection res;
  const int32 rank = dray::mpi_rank();
  const RankTasks &tasks = distribution[rank];

  const int32 num_tasks = tasks.m_tasks.size();
  std::vector<int32> local_dests;
  for(int32 i = 0; i < num_tasks; ++i)
  {
    DataSet dataset = collection.domain(i);
    const DomainTask &task = tasks.m_tasks[i];
    if(task.m_splits.size() > 0)
    {
      detail::split(task, dataset, res, local_dests);
    }
    else
    {
      // no splits just pass through the data
      res.add_domain(dataset);
      local_dests.push_back(rank);
    }
  }
#ifdef DRAY_MPI_ENABLED
  int32 local_doms = res.local_size();

  if(local_dests.size() != local_doms)
  {
    DRAY_ERROR("dests and doms mismatch");
  }

  MPI_Comm mpi_comm = MPI_Comm_f2c(dray::mpi_comm());
  const int32 comm_size = dray::mpi_size();
  const int32 global_size = res.size();

  std::vector<int32> global_counts;
  global_counts.resize(comm_size);
  MPI_Allgather(&local_doms, 1, MPI_INT, &global_counts[0], 1, MPI_INT, mpi_comm);

  std::vector<int32> global_offsets;
  global_offsets.resize(comm_size);
  global_offsets[0] = 0;

  for(int i = 1; i < comm_size; ++i)
  {
    global_offsets[i] = global_offsets[i-1] + global_counts[i-1];
  }

  dest_list.resize(global_size);

  MPI_Allgatherv(&local_dests[0],
                 local_doms,
                 MPI_INT,
                 &dest_list[0],
                 &global_counts[0],
                 &global_offsets[0],
                 MPI_INT,
                 mpi_comm);
  src_list.resize(global_size);
  for(int32 i = 0; i < comm_size; ++i)
  {
    const int32 start = global_offsets[i];
    for(int32 c = 0; c < global_counts[i]; ++c)
    {
      src_list[start + c] = i;
    }
  }
#endif
  return res;
}

void
VolumeBalance::map(const std::vector<RankTasks> &distribution,
                   std::vector<int32> &src_list,
                   std::vector<int32> &dest_list)
{
  const int32 size = distribution.size();
  for(int32 i = 0; i < size; ++i)
  {
    const RankTasks &tasks = distribution[i];
    const int32 num_tasks = tasks.m_tasks.size();
    for(int32 task_idx = 0; task_idx < num_tasks; ++task_idx)
    {
      // one part stays on this rank
      src_list.push_back(i);
      dest_list.push_back(i);
      for(auto &split : tasks.m_tasks[task_idx].m_splits)
      {
        src_list.push_back(i);
        dest_list.push_back(split.second);
      }
    }
  }
}

}//namespace dray

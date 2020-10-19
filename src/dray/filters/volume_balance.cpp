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

void split(const DomainTask &task, DataSet &dataset, Collection &col)
{
  DRAY_LOG_OPEN("split");
  AABB<3> bounds = dataset.topology()->bounds();
  const int32 max_comp = bounds.max_dim();
  float32 length = bounds.m_ranges[max_comp].length();

  float32 total = task.m_amount;
  const int32 pieces = task.m_splits.size();
  for(int32 i = 0; i < pieces; ++i)
  {
    total += task.m_splits[i].first;
  }
  std::vector<float32> divisions;
  divisions.resize(pieces+1);
  divisions[0] = (task.m_amount / total) * length;

  for(int32 i = 0; i < pieces; ++i)
  {
    divisions[i+1] = (task.m_splits[i].first / total) * length;
  }

  std::vector<float32> ranges;
  ranges.resize(divisions.size() + 1);

  ranges[0] = bounds.m_ranges[max_comp].min();
  ranges[ranges.size() - 1] = bounds.m_ranges[max_comp].max() + length * 1e-3;

  for(int32 i = 0; i < divisions.size() - 1; ++i)
  {
    ranges[i+1] = ranges[i] + divisions[i];
    std::cout<<"["<<dray::mpi_rank()<<"]: "<<i+1
             <<" range "<<ranges[i]<<" + "<<divisions[i]<<" = "<<ranges[i+1]<<"\n";
  }

  std::cout<<"["<<dray::mpi_rank()<<"]: ranges "<<ranges.size()<<"\n";
  for(int i = 0; i < ranges.size(); ++i)
  {
    std::cout<<"["<<dray::mpi_rank()<<"]: range "<<i<<" "<<ranges[i]<<"\n";
  }

  for(int32 i = 0; i < ranges.size() - 1; ++i)
  {
    MaskFunctor func(max_comp,ranges[i],ranges[i+1]);
    dispatch(dataset.topology(), func);
    Subset subset;
    DataSet piece = subset.execute(dataset, func.m_mask);
    std::cout<<"["<<dray::mpi_rank()<<"]: "
             <<" splitting on range "<<ranges[i]<<" - "<<ranges[i+1]
             <<" elements "<<piece.topology()->cells()<<"\n";
    col.add_domain(piece);
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
  std::cout<<"ave "<<ave<<"\n";

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
VolumeBalance::execute(Collection &collection)
{
  Collection res;

  const int32 local_doms = collection.local_size();
  std::vector<float32> local_volumes;
  local_volumes.resize(local_doms);

  float32 total_volume = 0;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet dataset = collection.domain(i);
    local_volumes[i] = dataset.topology()->bounds().volume();
    total_volume += local_volumes[i];
  }


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
  std::cout<<"Ratio "<<ratio<<"\n";
  if(rank == 0)
  {
    //std::cout<<"plan:\n";
    //for(int i = 0; i < plan.size(); ++i)
    //{
    //  std::cout<<" src "<<plan[i].m_src<<"\n";
    //  std::cout<<" dest "<<plan[i].m_dest<<"\n";
    //  std::cout<<" amount "<<plan[i].m_amount<<"\n";
    //}
    for(int i = 1; i < global_size; ++i)
    {
      std::cout<<"Index "<<i<<" "<<global_volumes[i]<<"\n";
    }
  }
  Collection chopped = chopper(distribution, collection);

  for(int i = 0; i < chopped.local_size(); ++i)
  {
    std::cout<<"["<<dray::mpi_rank()<<"]: chopped field range "<<i
    <<" "<<chopped.domain(i).field("density")->range()[0] <<"\n";
  }

  std::vector<int32> src_list;
  std::vector<int32> dest_list;
  map(distribution, src_list, dest_list);

  std::cout<<"Chopped size "<<chopped.size()<<"\n";
  std::cout<<"src size "<<src_list.size()<<"\n";


  Redistribute redist;
  res = redist.execute(chopped, src_list, dest_list);
  for(int i = 0; i < res.local_size(); ++i)
  {
    std::cout<<"["<<dray::mpi_rank()<<"]: res field range "<<i
    <<" "<<res.domain(i).field("density")->range()[0] <<"\n";
  }
#endif
  return res;
}

Collection
VolumeBalance::chopper(const std::vector<RankTasks> &distribution,
                       Collection &collection)
{
  Collection res;
  const int32 rank = dray::mpi_rank();
  const RankTasks &tasks = distribution[rank];

  const int32 num_tasks = tasks.m_tasks.size();
  for(int32 i = 0; i < num_tasks; ++i)
  {
    DataSet dataset = collection.domain(i);
    const DomainTask &task = tasks.m_tasks[i];
    if(task.m_splits.size() > 0)
    {
      detail::split(task, dataset, res);
    }
    else
    {
      // no splits just pass through the data
      res.add_domain(dataset);
    }
  }

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

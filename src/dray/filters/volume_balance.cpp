#include <dray/filters/volume_balance.hpp>

#include <dray/dispatcher.hpp>
#include <dray/dray.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/error_check.hpp>
#include <numeric>
#include <algorithm>

#ifdef DRAY_MPI_ENABLED
#include<mpi.h>
#endif

namespace dray
{

namespace detail
{


}//namespace detail

VolumeBalance::VolumeBalance()
{
}

float32
VolumeBalance::perfect_splitting(std::vector<float32> volumes,
                                 std::map<int32,std::vector<Task>> &plan)
{
  plan.clear();
  const int32 size = volumes.size();

  std::vector<int32> idx(size);
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
       [&volumes](int32 i1, int32 i2) {return volumes[i1] < volumes[i2];});

  float32 sum = 0;
  for(int32 i = 0; i < size; ++i)
  {
    sum += volumes[i];
  }
  const float32 ave = sum / float32(size);
  std::cout<<"ave "<<ave<<"\n";

  int32 giver = size - 1;
  int32 taker = 0;
  float32 eps = volumes[idx[giver]] * 1e-3;
  float32 max_val = volumes[idx[giver]];
  //std::cout<<"Taker "<<taker<<"\n";
  while(giver > taker)
  {
    //std::cout<<"Giver "<<giver<<" taker "<<taker<<"\n";
    int32 giver_idx = idx[giver];
    int32 taker_idx = idx[taker];
    float32 giver_work = volumes[giver_idx];
    if(giver_work < ave)
    {
      break;
    }

    float32 giver_dist = giver_work - ave;

    float32 taker_dist = ave - volumes[taker_idx];
    float32 give_amount = 0;
    if(taker_dist > giver_dist)
    {
      give_amount = giver_dist;
    }
    else
    {
      give_amount = taker_dist;
    }

    if(give_amount >  eps)
    {
      Task task;
      //task.m_src = giver_idx;
      task.m_dest = taker_idx;
      task.m_amount = give_amount;
      plan[giver_idx].push_back(task);
    }

    volumes[taker_idx] += give_amount;
    volumes[giver_idx] -= give_amount;
    //std::cout<<"giver "<<volumes[giver_idx]<<" taker "<<volumes[taker_idx]<<"\n";

    // does giver have more?
    if(volumes[giver_idx] <= ave + eps)
    {
      giver--;
    }
    // can taker take more?
    if(volumes[taker_idx] >= ave - eps)
    {
      taker++;
    }
  }
  //std::cout<<"giver "<<giver<<" taker "<<taker<<"\n";

  float32 max_after = 0;
  for(int32 i = 0; i < size; ++i)
  {
    max_after = std::max(volumes[i], max_after);
  }

  if(dray::mpi_rank() == 0)
  {
    std::ofstream load_file;
    load_file.open ("perfect_splits.txt");
    for(auto l : volumes)
    {
      load_file << l <<"\n";
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


  std::map<int32,std::vector<Task>> plan;
  float32 ratio = perfect_splitting(rank_volumes, plan);
  std::cout<<"Ratio "<<ratio<<" plan_size "<<plan.size()<<"\n";
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
  Collection chopped = chopper(collection, plan[rank]);
#endif
  return res;
}


}//namespace dray

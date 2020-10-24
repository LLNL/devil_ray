#ifndef DRAY_VOLUME_BALANCE_HPP
#define DRAY_VOLUME_BALANCE_HPP

#include <dray/collection.hpp>
#include <dray/rendering/camera.hpp>

namespace dray
{

struct DomainTask
{
  float32 m_amount;
  // what was chopped off and where is it going
  std::vector<std::pair<float32, int32>> m_splits;
  void chunk(float32 amount, int32 dest_rank)
  {
    std::pair<float32, int32> split;
    split.first = amount;
    split.second = dest_rank;
    m_amount -= amount;
    m_splits.push_back(split);

    if(m_amount <= 0)
    {
      std::cout<<"task that was split now <=0\n";
    }
  }
};

struct RankTasks
{
  std::vector<DomainTask> m_tasks;
  float32 m_total_amount;
  int32 m_rank;

  void add_task(const float32 amount)
  {
    DomainTask task;
    task.m_amount = amount;
    m_total_amount += amount;
    m_tasks.push_back(task);
  }

  float32 amount() const
  {
    return m_total_amount;
  }

  void add(const float32 amount)
  {
    m_total_amount += amount;
  }

  int32 biggest()
  {
    int32 max_index = -1;
    float32 max_amount = -1.f;

    const int32 size = m_tasks.size();
    for(int32 i = 0; i < size; ++i)
    {
      if(m_tasks[i].m_amount > max_amount)
      {
        max_index = i;
        max_amount = m_tasks[i].m_amount;
      }
    }
    return max_index;
  }

  float32 split_biggest(const float32 amount, const int32 dest_rank)
  {
    int32 idx = biggest();
    if(idx == -1) std::cout<<"bad\n";

    m_tasks[idx].chunk(amount, dest_rank);

    m_total_amount -= amount;

    if(m_total_amount <= 0)
    {
      std::cout<<"Rank task total now <=0\n";
    }
    // future: this could return
    return amount;
  }
};

class VolumeBalance
{
protected:
public:
  VolumeBalance();
  Collection execute(Collection &collection, Camera &camera);
  Collection execute2(Collection &collection, Camera &camera);

  float32 perfect_splitting(std::vector<RankTasks> &distribution);

  float32 schedule_blocks(std::vector<float32> &rank_volumes,
                          std::vector<int32> &global_counts,
                          std::vector<int32> &global_offsets,
                          std::vector<float32> &global_volumes,
                          std::vector<int32> &src_list,
                          std::vector<int32> &dest_list);

  float32 schedule_blocks2(std::vector<float32> &rank_volumes,
                           std::vector<int32> &global_counts,
                           std::vector<int32> &global_offsets,
                           std::vector<float32> &global_volumes,
                           std::vector<int32> &src_list,
                           std::vector<int32> &dest_list);

  Collection chopper(const std::vector<RankTasks> &distribution,
                     Collection &collection,
                     std::vector<int32> &src_list,
                     std::vector<int32> &dest_list);

  Collection chopper(float32 piece_size,
                     std::vector<float32> &sizes,
                     Collection &collection);

  void allgather(std::vector<float32> &local_volumes,
                 const int32 global_size,
                 std::vector<float32> &rank_volumes,
                 std::vector<int32> &global_counts,
                 std::vector<int32> &global_offsets,
                 std::vector<float32> &global_volumes);

  float32 volumes(Collection &collection,
                  Camera &camera,
                  std::vector<float32> &volumes);

  void map(const std::vector<RankTasks> &distribution,
           std::vector<int32> &src_list,
           std::vector<int32> &dest_list);
};

};//namespace dray

#endif

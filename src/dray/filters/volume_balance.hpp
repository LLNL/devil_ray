#ifndef DRAY_VOLUME_BALANCE_HPP
#define DRAY_VOLUME_BALANCE_HPP

#include <dray/collection.hpp>

namespace dray
{

struct DomainTask
{
  float32 m_amount;
  // what was chopped off and where is it going
  std::vector<std::pair<float32, int32>> m_splits;
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

    std::pair<float32, int32> split;
    split.first = amount;
    split.second = dest_rank;
    m_tasks[idx].m_splits.push_back(split);

    m_total_amount -= amount;

    if(m_tasks[idx].m_amount <= 0)
    {
      std::cout<<"task that was split now <=0\n";
    }
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
  Collection execute(Collection &collection);
  float32 perfect_splitting(std::vector<RankTasks> &distribution);
  Collection chopper(const std::vector<RankTasks> &distribution,
                     Collection &collection,
                     std::vector<int32> &src_list,
                     std::vector<int32> &dest_list);

  void map(const std::vector<RankTasks> &distribution,
           std::vector<int32> &src_list,
           std::vector<int32> &dest_list);
};

};//namespace dray

#endif

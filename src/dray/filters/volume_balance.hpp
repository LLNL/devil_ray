#ifndef DRAY_VOLUME_BALANCE_HPP
#define DRAY_VOLUME_BALANCE_HPP

#include <dray/collection.hpp>

namespace dray
{

struct DomainTask
{
  int32 m_src_rank;
  float32 m_amount;
};

class VolumeBalance
{
protected:
public:
  VolumeBalance();
  Collection execute(Collection &collection);
  float32 perfect_splitting(std::vector<float32> volume,
                            std::map<int32,std::vector<Task>> &plan);
  void chopper(std::vector<float32> &local_volumes, const std::vector<Task> &plan)
};

};//namespace dray

#endif

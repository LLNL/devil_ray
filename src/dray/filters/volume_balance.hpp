#ifndef DRAY_VOLUME_BALANCE_HPP
#define DRAY_VOLUME_BALANCE_HPP

#include <dray/collection.hpp>
#include <dray/rendering/camera.hpp>

namespace dray
{

class VolumeBalance
{
protected:
  bool m_use_prefix;
  int32 m_piece_factor;
public:
  VolumeBalance();

  void prefix_balancing(bool on);
  void piece_factor(int32 size);

  Collection execute(Collection &collection, Camera &camera);

  float32 schedule_blocks(std::vector<float32> &rank_volumes,
                          std::vector<int32> &global_counts,
                          std::vector<int32> &global_offsets,
                          std::vector<float32> &global_volumes,
                          std::vector<int32> &src_list,
                          std::vector<int32> &dest_list);

  float32 schedule_prefix(std::vector<float32> &rank_volumes,
                          std::vector<int32> &global_counts,
                          std::vector<int32> &global_offsets,
                          std::vector<float32> &global_volumes,
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

};

};//namespace dray

#endif

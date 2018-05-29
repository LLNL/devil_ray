#ifndef DRAY_RAY_HPP
#define DRAY_RAY_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

template<typename T>
class Ray
{
public:
  Array<Vec<T,3>> m_dir;
  Array<Vec<T,3>> m_orig;
  Array<T>        m_near;
  Array<T>        m_far;
  Array<T>        m_dist;
  Array<int32>    m_pixel_id;
  Array<int32>    m_hit_idx;
  void resize(const int32 size);
  int32 size() const;
};

typedef Ray<float32> ray32;
typedef Ray<float64> ray64;

} // namespace dray
#endif

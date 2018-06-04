#ifndef DRAY_POINT_LOCATION_HPP
#define DRAY_POINT_LOCATION_HPP

#include <dray/array.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/vec.hpp>

namespace dray
{

class PointLocator
{
protected:
  BVH             m_bvh;

  PointLocator(); 
public:
  PointLocator(BVH bvh); 
  ~PointLocator(); 
  
  template<typename T>
  void            locate_candidates(Array<Vec<T, 3>> &points);

};

} // namespace dray

#endif

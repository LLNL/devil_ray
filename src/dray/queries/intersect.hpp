#ifndef DRAY_INTERSECT_HPP
#define DRAY_INTERSECT_HPP

#include <dray/collection.hpp>
#include <vector>

namespace dray
{

class Intersect
{
public:
  Intersect();
  void execute(Collection &collection,
               const std::vector<Vec<float64,3>> &directions, //world space
               const std::vector<Vec<float64,3>> &ips,        //reference space
               Array<int32> &face_ids,           // not sure we need this
               Array<Vec<Float,3>> &res_ips);   //
};

};//namespace dray

#endif//DRAY_INTERSECT_HPP

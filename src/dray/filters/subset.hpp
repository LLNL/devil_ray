#ifndef DRAY_SUBSET_HPP
#define DRAY_SUBSET_HPP

#include <dray/collection.hpp>

namespace dray
{

class Subset
{
public:
  Subset();
  DataSet execute(DataSet &dataset, Array<int32> &cell_mask);
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP

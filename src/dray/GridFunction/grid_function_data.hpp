#ifndef DRAY_GRID_FUNCTION_DATA_HPP
#define DRAY_GRID_FUNCTION_DATA_HPP

#include <dray/el_trans.hpp>

namespace dray
{
// TODO consolidate ElTransData to here.
template <int32 PhysDim> using GridFunctionData = ElTransData<PhysDim>;
// TODO int32 get_num_elem()
} // namespace dray

#endif // DRAY_GRID_FUNCTION_DATA_HPP

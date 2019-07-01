#ifndef DRAY_GET_SHADING_CONTEXT_HPP
#define DRAY_GET_SHADING_CONTEXT_HPP

#include <dray/data_set.hpp>
#include <dray/shading_context.hpp>
#include <dray/ray.hpp>

// Ultimately, this file will not be installed with dray
// PRIVATE in cmake
namespace dray
{
namespace internal
{

template <typename T>
Array<ShadingContext<T>>
get_shading_context(Array<Ray<T>> &rays,
                    Field<T> &field,
                    Mesh<T> &mesh,
                    Array<RefPoint<T,3>> &rpoints);

}; // namespace internal

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP

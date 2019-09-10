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

template <typename T, class ElemT>
Array<ShadingContext<T>>
get_shading_context(Array<Ray<T>> &rays,
                    Field<T, FieldOn<ElemT, 1u>> &field,
                    Mesh<T, ElemT> &mesh,
                    Array<RefPoint<T, ElemT::get_dim()>> &rpoints);

/// template <typename T, class ElemT>
/// Array<ShadingContext<T>>
/// get_shading_context(Array<Ray<T>> &rays,
///                     Array<Vec<int32,2>> &faces,
///                     Field<T, FieldOn<ElemT, 1u>> &field,
///                     Mesh<T, ElemT> &mesh,
///                     Array<RefPoint<T,3>> &rpoints);

}; // namespace internal

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP

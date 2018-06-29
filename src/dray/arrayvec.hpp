#ifndef DRAY_ARRAYVEC_HPP
#define DRAY_ARRAYVEC_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/matrix.hpp>

namespace dray
{

//
// ScalarVec: Template that is either a scalar or a Vec.
//
template<typename T, int S> struct _ScalarVec { typedef Vec<T,S> type; };
template<typename T> struct _ScalarVec<T,1> { typedef T type; };

template <typename T, int S>
using ScalarVec = typename _ScalarVec<T,S>::type;

//
// ArrayVec
//
template <typename T, int S>
using ArrayVec = Array<typename _ScalarVec<T,S>::type>;

} // namespace dray

#endif

#ifndef DRAY_ARRAYVEC_HPP
#define DRAY_ARRAYVEC_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/matrix.hpp>

namespace dray
{

namespace detail
{
// Template that is either a scalar or a Vec.
template<typename T, int S> struct _ScalarVec { typedef Vec<T,S> type; };
template<typename T> struct _ScalarVec<T,1> { typedef T type; };

//// Template that includes both Arrays over scalars and Arrays over Vec.
//template <typename T, int S>
//struct _ArrayVec
//{
//  typedef Array<typename detail::ScalarVec<T,S>::type> type;
//};
}  // namespace detail

template <typename T, int S>
using ScalarVec = typename detail::_ScalarVec<T,S>::type;

template <typename T, int S>
using ArrayVec = Array<typename detail::_ScalarVec<T,S>::type>;

//TODO ArrayMatrix
template <typename T, int R, int C>
using ArrayMatrix = Array<Matrix<T,R,C>>;

} // namespace dray

#endif

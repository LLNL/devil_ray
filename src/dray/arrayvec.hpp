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
// ScalarMatrix: Template that is either a scalar or a Vec or a Matrix.
//
template<typename T, int R, int C>  struct _ScalarMatrix { typedef Matrix<T,R,C> type; };
template<typename T, int R>  struct _ScalarMatrix<T,R,1> { typedef ScalarVec<T,R> type; };
template<typename T, int C>  struct _ScalarMatrix<T,1,C> { typedef ScalarVec<T,C> type; };

template <typename T, int R, int C>
using ScalarMatrix = typename _ScalarMatrix<T,R,C>::type;


//
// ArrayVec
//
template <typename T, int S>
using ArrayVec = Array<typename _ScalarVec<T,S>::type>;


//
// ArrayMatrix
//
template <typename T, int R, int C>
using ArrayMatrix = Array<typename _ScalarMatrix<T,R,C>::type>;

} // namespace dray

#endif

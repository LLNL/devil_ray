#ifndef DRAY_ARRAYVEC_HPP
#define DRAY_ARRAYVEC_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>

namespace dray
{

namespace detail
{
// Template that is either a scalar or a Vec.
template<typename T, int S> struct ScalarVec { typedef Vec<T,S> type; };
template<typename T> struct ScalarVec<T,1> { typedef T type; };
}  // namespace detail

// Template that includes both Arrays over scalars and Arrays over Vec.
template <typename T, int S>
struct ArrayVec
{
  typedef Array<typename detail::ScalarVec<T,S>::type> type;
};

} // namespace dray

#endif

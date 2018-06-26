#ifndef DRAY_HIGH_ORDER_SHAPE_HPP
#define DRAY_HIGH_ORDER_SHAPE_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/math.hpp>
#include <dray/types.hpp>

#include <assert.h>


//
//TODO
//  - Redesign for special case of scalar field, so no Vec<T,1> in array.

namespace dray
{

/// // Value template indicating that a shape functor
/// // has D input components, DOF output components.
/// template <int32 D, int32 DOF>
/// struct ShapeDims {};

namespace detail
{

// Bernstein P-order, D-dimensional evaluator
template <typename T, int32 D, int32 P>
struct BernsteinShape
{
  static constexpr int32 DOF = IntPow<P+1,D>::val;
  /// static const ShapeDims<D,DOF> shape_dims;

  DRAY_EXEC void operator()(const Vec<T,D> &ref_pt, Vec<T,DOF> &shape_out) const
  ///void operator()(const Vec<T,D> &ref_pt, Vec<T,DOF> &shape_out) const
  {
    //TODO
  }
};

} // namespace detail


  // DOF -- #degrees of freedom i.e. components of the shape tuple.
  // D   -- #dimensions of input
template <typename T, typename FunctionShape1, typename FunctionShape2>
struct PairShape
{
public:
  FunctionShape1 f1;
  FunctionShape2 f2;
};

  // C   -- #components of output
  // DOF -- #control points per element
template <typename T, int32 C, int32 DOF>
class FunctionCtrlPoints
{
//private: //TODO
public:
  // There is size_elt == # elements.
  // and there is size_ctrl == total # control points.

  // Based on the MFEM GridFunction.

  // Functions are represented as a linear combination of basis functions over the unit hypercube.
  // For a given element, the values of the control points are coefficients to the basis functions.
  //
  // The basis functions are joined into a tuple function called a "shape function."
  // The element function and its derivative can be computed by evaluating the shape function
  // and taking dot product with the element values.
  //
  // Neighboring elements may share some control points.

  Array<int32> m_ctrl_idx;    // 0 <= ii < size_elt, 0 <= jj < DOF, 0 <= m_ctrl_idx[ii*DOF + jj] < size_ctrl
  Array<Vec<T,C>> m_values;   // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

    // D  -- #intrinsic dimensions, i.e. number of inputs to the shape function.
  template<typename ShapeFunctor, int32 D>
  Array<Vec<T,C>> eval(const ShapeFunctor &_shape_f, const Array<Vec<T,D>> &ref_pts);

};
// 

//template<typename FunctionShape>
//class 

} // namespace dray

#endif

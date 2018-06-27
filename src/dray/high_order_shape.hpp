#ifndef DRAY_HIGH_ORDER_SHAPE_HPP
#define DRAY_HIGH_ORDER_SHAPE_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/arrayvec.hpp>  // Template Array<T> or Array<Vec<T,S>>
#include <dray/math.hpp>
#include <dray/types.hpp>


namespace dray
{

// --- Shape Type Public Interface --- //
//
// int32 get_el_dofs() const;
// int32 get_ref_dim() const;
// void calc_shape_dshape(const Array<int> &active_idx, const ArrayVec<RefDim> &ref_pts, Array<T> &shape_val, ArrayVec<RefDim> &shape_deriv) const; 
//
//
// --- Internal Parameters (example) --- //
//
// static constexpr int32 ref_dim;
// int32 p_order;
// int32 el_dofs;

// Tensor product of arrays. Input arrays start at starts[0], starts[1], ... and each has stride of Stride.
// (This makes array storage flexible: they can be stored separately, contiguously, or interleaved.)
// The layout of the output array is such that the last given index is iterated first (innermost, stride of 1),
// and the first given index is iterated last (outermost, stride of el_dofs_1d ^ (RefDim-1)).
template <typename T, int32 RefDim, int32 InStride=1>
struct TensorProduct
{
  // Computes and stores a single component of the tensor.
  DRAY_EXEC
  void operator() (int32 el_dofs_1d, const T* starts[], int32 out_idx, T* out) const
  {
    //int32 out_stride = 1;
    //int32 out_idx = 0;
    T out_val = 1;
    for (int32 rdim = RefDim-1, idx_mask_right = 1; rdim >= 0; rdim--, idx_mask_right *= el_dofs_1d)
    {
      //int32 dim_idx = idx[rdim];
      int32 dim_idx = (out_idx / idx_mask_right) % el_dofs_1d;
      //out_idx += dim_idx * dim_stride;
      out_val *= starts[rdim][dim_idx * InStride];
    }
    out[out_idx] = out_val;
  }
};

/// template <typename T, int32 RefDim>
/// struct BernsteinShape
/// {
///   int32 p_order;
/// 
///   int32 get_el_dofs() const { return pow(p_order + 1, RefDim); }
///   int32 get_ref_dim() const { return RefDim; }
/// 
///   void calc_shape_dshape(const Array<int32> &active_idx,
///                          const ArrayVec<RefDim> &ref_pts,
///                          Array<T> &shape_val,
///                          ArrayVec<RefDim> &shape_deriv) const;
/// 
/// protected:
///   DRAY_EXEC
///   static void calc_shape_dshape_1d(const 
/// };

//  v---- USE THIS
//template <typename T>
//struct BernsteinShape_Internals
//{
//  // Bernstein evaluator rippped out of MFEM.
//  DRAY_EXEC
//  static void calc_shape_dshape_1d(int32 el_dofs_1d, const T x, const T y, T *u, T *d)
//  {
//    const int32 p = el_dofs_1d - 1;
//    if (p == 0)
//    {
//       u[0] = 1.;
//       d[0] = 0.;
//    }
//    else
//    {
//       int i;
//       const int *b = Binom(p);
//       const double xpy = x + y, ptx = p*x;
//       double z = 1.;
//
//       for (i = 1; i < p; i++)
//       {
//          d[i] = b[i]*z*(i*xpy - ptx);
//          z *= x;
//          u[i] = b[i]*z;
//       }
//       d[p] = p*z;
//       u[p] = z*x;
//       z = 1.;
//       for (i--; i > 0; i--)
//       {
//          d[i] *= z;
//          z *= y;
//          u[i] *= z;
//       }
//       d[0] = -p*z;
//       u[0] = z*y;
//    }
//  }
//};

// Abstract class defines mechanics of tensor product.
// Inherit from this and define 2 methods:
// 1. Class method get_el_dofs_1d();
// 2. External function in the (template parameter) class Shape1D, calc_shape_dshape_1d();
template <typename T, int32 RefDim, typename Shape1D>
struct TensorShape
{
  virtual int32 get_el_dofs_1d() const = 0;

  int32 get_el_dofs() const { return pow(get_el_dofs_1d(), RefDim); }

  void calc_shape_dshape( const Array<int32> &active_idx,
                          const ArrayVec<T,RefDim> &ref_pts,
                          Array<T> &shape_val,                      // Will be resized.
                          ArrayVec<T,RefDim> &shape_deriv) const;   // Will be resized.
};

template <typename T>
struct Linear1D
{
  DRAY_EXEC
  static void calc_shape_dshape_1d(int32 eldofs_1d, const T x, const T y, T *u, T *d)
  {
    // Note that the parameter eldofs_1d is disregarded.
    // For linear elements it is supposed to be 2.

    // Linear interpolation between two eldofs.
    u[0] = y;
    u[1] = x;
    d[0] = -1;
    d[1] = 1;
  }
};

template <typename T, int32 RefDim>
struct LinearShape : public TensorShape<T, RefDim, Linear1D<T>>
{
  virtual int32 get_el_dofs_1d() const { return 2; }

  //int32 get_el_dofs() const { return 2 >> (RefDim - 1); }
  //int32 get_ref_dim() const { return RefDim; }

  // Inherits from TensorShape
  // - int32 get_el_dofs() const;
  // - void calc_shape_dshape(...) const;
};


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
  // D   -- #intrinsic dimensions, i.e. number of inputs to the shape function.
template <typename T, int32 C, int32 DOF, typename ShapeFunctor, int32 D>
class FunctionCtrlPoints
{
//private: //TODO
public:

  typedef ScalarVec<T,C> PhysVec;
  typedef ScalarVec<T,DOF> ShapeVec;
  typedef ScalarVec<T,D> RefVec;

  // A shape functor might not have the same ShapeVec type, but it should.
  // Otherwise, eval() and eval_d() will generate compiler errors.

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
  Array<PhysVec> m_values;   // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

  Array<PhysVec> eval(const ShapeFunctor &_shape_f, const Array<RefVec> &ref_pts) const;

};


// 

//template<typename FunctionShape>
//class 

} // namespace dray

#endif




// ///////// //
// THE ABYSS //
// ///////// //


/// namespace detail
/// {
/// // Discrete uniform distribution.
/// template <typename T, int32 D, int32 DOF>
/// struct DummyUniformShape
/// {
///   // Evaluates basis functions.
///   void calc_shape(const dray::Vec<T,D> &ref_pt, dray::Vec<T,DOF> &shape_out) const
///   {
///     shape_out = static_cast<T>(1.f) / DOF;
///   }
/// 
///   /// // Evaluates derivatives of basis functions.
///   /// void calc_d_shape(const dray::Vec<T,D> &ref_pt, dray::Vec<T,DOF> &d_shape_out) const
///   /// {
///   ///   d_shape_out = static_cast<T>(0.f);
///   /// }
/// };
/// 
/// // Bernstein P-order, D-dimensional evaluator
/// template <typename T, int32 D, int32 P>
/// struct BernsteinShape
/// {
///   static constexpr int32 DOF = IntPow<P+1,D>::val;
///   /// static const ShapeDims<D,DOF> shape_dims;
/// 
///   typedef ScalarVec<T,D> RefVec;
///   typedef ScalarVec<T,DOF> ShapeVec;
/// 
///   DRAY_EXEC void calc_shape(const RefVec &ref_pt, ShapeVec &shape_out) const
///   {
///     //TODO
///   }
/// 
///   ////DRAY_EXEC void calc_d_shape(const RefVec &ref_pt, ShapeVec &d_shape_out) const
///   ////{
///   ////  //TODO
///   ////}
/// };
/// 
/// 
/// } // namespace detail



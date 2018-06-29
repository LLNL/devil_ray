#ifndef DRAY_HIGH_ORDER_SHAPE_HPP
#define DRAY_HIGH_ORDER_SHAPE_HPP

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/arrayvec.hpp>  // Template Array<T> or Array<Vec<T,S>>
#include <dray/binomial.hpp>
#include <dray/math.hpp>
#include <dray/types.hpp>

#include <stddef.h>


namespace dray
{

// Tensor product of arrays. Input arrays start at starts[0], starts[1], ... and each has stride of Stride.
// (This makes array storage flexible: they can be stored separately, contiguously, or interleaved.)
// The layout of the output array is such that the last given index is iterated first (innermost, stride of 1),
// and the first given index is iterated last (outermost, stride of el_dofs_1d ^ (RefDim-1)).
template <typename T, int32 RefDim, int32 InStride, int32 OutStride>
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
    out[out_idx * OutStride] = out_val;
  }
};


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


// Abstract TensorShape class defines mechanics of tensor product.
// Inherit from this and define 2 methods:
// 1. Class method get_el_dofs_1d();
// 2. External function in the (template parameter) class Shape1D, calc_shape_dshape_1d();
template <typename T, int32 RefDim, typename Shape1D>
struct TensorShape
{
  virtual int32 get_el_dofs_1d() const = 0;

  int32 get_ref_dim() const { return RefDim; }
  int32 get_el_dofs() const { return pow(get_el_dofs_1d(), RefDim); }

  void calc_shape_dshape( const Array<int32> &active_idx,
                          const ArrayVec<T,RefDim> &ref_pts,
                          Array<T> &shape_val,                      // Will be resized.
                          ArrayVec<T,RefDim> &shape_deriv) const;   // Will be resized.
};

template <typename T>
struct Bernstein1D
{
  // Bernstein evaluator rippped out of MFEM.
  DRAY_EXEC
  static void calc_shape_dshape_1d(int32 el_dofs_1d, const T x, const T y, T *u, T *d)
  {
    const int32 p = el_dofs_1d - 1;
    if (p == 0)
    {
       u[0] = 1.;
       d[0] = 0.;
    }
    else
    {
       // Write binomial coefficients into u memory instead of allocating b[].
       BinomRow<T>::fill_single_row(p,u);

       const double xpy = x + y, ptx = p*x;
       double z = 1.;

       int i;
       for (i = 1; i < p; i++)
       {
          //d[i] = b[i]*z*(i*xpy - ptx);
          d[i] = u[i]*z*(i*xpy - ptx);
          z *= x;
          //u[i] = b[i]*z;
          u[i] = u[i]*z;
       }
       d[p] = p*z;
       u[p] = z*x;
       z = 1.;
       for (i--; i > 0; i--)
       {
          d[i] *= z;
          z *= y;
          u[i] *= z;
       }
       d[0] = -p*z;
       u[0] = z*y;
    }
  }
};

template <typename T, int32 RefDim>
struct BernsteinShape : public TensorShape<T, RefDim, Bernstein1D<T>>
{
  int32 m_p_order;

  virtual int32 get_el_dofs_1d() const { return m_p_order + 1; }

  // Inherits from TensorShape
  // - int32 get_ref_dim() const;
  // - int32 get_el_dofs() const;
  // - void calc_shape_dshape(...) const;
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
class LinearShape : public TensorShape<T, RefDim, Linear1D<T>>
{
public:
  virtual int32 get_el_dofs_1d() const { return 2; }
  int32 get_ref_dim() const { return RefDim; }

  // Inherits from TensorShape
  // - int32 get_el_dofs() const;
  // - void calc_shape_dshape(...) const;
};


// Array "Fixed Size": Interface wrapper around Array with resize() removed.
template <typename T>
class ArrayFS
{
  Array<T> *a;
  const Array<T> *ac;
public:
  void set(Array<T> &other) { a = &other; ac = &other; }
  void set_const(const Array<T> &other) { a = NULL; ac = &other; }

  size_t size() const { return ac->size(); }
  T* get_host_ptr() { return a->get_host_ptr(); }
  T* get_device_ptr() { return a->get_device_ptr(); }
  const T* get_host_ptr_const() const { return ac->get_host_ptr_const(); }
  const T* get_device_ptr_const() const { return ac->get_device_ptr_const(); }
  void summary() { a->summary(); }
};


// Collection of element transformations from reference space to physical space.
template <typename T, int32 PhysDim, int32 RefDim, typename ShapeType>
class ElTrans
{
  // There is size_el == # elements.
  // and there is size_ctrl == total # control points.

  // Based on the MFEM GridFunction.

  // An element transformation is represented as a linear combination of basis functions over the unit hypercube.
  // For a given element, the values of the control points are coefficients to the basis functions.
  //
  // The basis functions are joined into a tuple function called a "shape function."
  // The element function and its derivative can be computed by evaluating the shape function
  // and taking dot product with the element values.
  //
  // Neighboring elements may share some control points.
private:
  int32 m_el_dofs;
  int32 m_size_el;
  int32 m_size_ctrl;
  ShapeType m_shape;
  Array<int32> m_ctrl_idx;    // 0 <= ii < size_el, 0 <= jj < el_dofs, 0 <= m_ctrl_idx[ii*el_dofs + jj] < size_ctrl
  ArrayVec<T,PhysDim> m_values;   // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

public:
  static constexpr int32 C_PhysDim = PhysDim;
  static constexpr int32 C_RefDim = RefDim;
  using C_ShapeType = ShapeType;

  void resize(int32 size_el, int32 el_dofs, ShapeType shape, int32 size_ctrl);

  // This method assumes that output arrays are already the right size.
  // It does not resize or assign new arrays to output parameters.
  void eval(const Array<int> &active_idx,
            const Array<int32> &el_ids, const ArrayVec<T,RefDim> &ref_pts,
            ArrayVec<T,PhysDim> &trans_val, Array<Matrix<T,PhysDim,RefDim>> &trans_deriv) const;

  // Clients may read and write contents of member arrays, but not size of member arrays.
  ArrayFS<int32>                      get_m_ctrl_idx()              { ArrayFS<int32> a; a.set(m_ctrl_idx); return a; }
  ArrayFS<ScalarVec<T,PhysDim>>       get_m_values()                { ArrayFS<ScalarVec<T,PhysDim>> a; a.set(m_values); return a; }
  const ArrayFS<int32>                get_m_ctrl_idx_const() const  { ArrayFS<int32> a; a.set_const(m_ctrl_idx); return a; }
  const ArrayFS<ScalarVec<T,PhysDim>> get_m_values_const()   const  { ArrayFS<ScalarVec<T,PhysDim>> a; a.set_const(m_values); return a; }

  /// const int32 *ctrl_idx_get_host_ptr_const() const;
  /// const int32 *ctrl_idx_get_device_ptr_const() const;
  /// int32 *ctrl_idx_get_host_ptr();
  /// int32 *ctrl_idx_get_device_ptr();

  /// const int32 *values_get_host_ptr_const() const;
  /// const int32 *values_get_device_ptr_const() const;
  /// int32 *values_get_host_ptr();
  /// int32 *values_get_device_ptr();
};

template <typename T, int32 PhysDim, int32 RefDim>
using ElTrans_BernsteinShape = ElTrans<T,PhysDim,RefDim, BernsteinShape<T,RefDim>>;


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



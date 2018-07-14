#ifndef DRAY_HIGH_ORDER_SHAPE_HPP
#define DRAY_HIGH_ORDER_SHAPE_HPP

#include <dray/ray.hpp>
#include <dray/array.hpp>
#include <dray/matrix.hpp>
#include <dray/vec.hpp>
#include <dray/binomial.hpp>
#include <dray/math.hpp>
#include <dray/types.hpp>

#include <dray/linear_bvh_builder.hpp>
#include <dray/aabb.hpp>
#include <dray/shading_context.hpp>

#include <stddef.h>


namespace dray
{
//// // ---------------------------------
//// // Support for Shape Type functions.
//// // ---------------------------------
//// 
//// // Tensor product of arrays. Input arrays start at starts[0], starts[1], ... and each has stride of Stride.
//// // (This makes array storage flexible: they can be stored separately, contiguously, or interleaved.)
//// // The layout of the output array is such that the last given index is iterated first (innermost, stride of 1),
//// // and the first given index is iterated last (outermost, stride of el_dofs_1d ^ (RefDim-1)).
//// template <typename T, int32 RefDim, int32 InStride, int32 OutStride>
//// struct TensorProduct
//// {
////   // Computes and stores a single component of the tensor.
////   DRAY_EXEC
////   void operator() (int32 el_dofs_1d, const T* starts[], int32 out_idx, T* out) const
////   {
////     //int32 out_stride = 1;
////     //int32 out_idx = 0;
////     T out_val = 1;
////     for (int32 rdim = RefDim-1, idx_mask_right = 1; rdim >= 0; rdim--, idx_mask_right *= el_dofs_1d)
////     {
////       //int32 dim_idx = idx[rdim];
////       int32 dim_idx = (out_idx / idx_mask_right) % el_dofs_1d;
////       //out_idx += dim_idx * dim_stride;
////       out_val *= starts[rdim][dim_idx * InStride];
////     }
////     out[out_idx * OutStride] = out_val;
////   }
//// };
//// 

namespace detail
{
template <typename T>
struct MultInPlace
{
  DRAY_EXEC static void mult(T *arr, T fac, int32 len)
  {
    for (int32 ii = 0; ii < len; ii++)
      arr[ii] *= fac;
  }
};

template <typename T, int32 S>
struct MultInPlace<Vec<T,S>>
{
  DRAY_EXEC static void mult(Vec<T,S> *arr, Vec<T,S> fac, int32 len)
  {
    for (int32 ii = 0; ii < len; ii++)
      for (int32 c = 0; c < S; c++)
        arr[ii][c] *= fac[c];
  }
};
} //  namespace detail

template <typename T>
struct SimpleTensor   // This means single product of some number of vectors.
{
  // Methods to compute a tensor from several vectors, in place.

  int32 s;   // Be sure to initialize this before using.

  DRAY_EXEC int32 get_size_tensor(int32 t_order) { return pow(s,t_order); }

  // Where you should store the vectors that will be used to construct the tensor.
  // First pointer is for the last/outermost looping index variable in the tensor: X1(Y1Y2Y3)X2(Y1Y2Y3)X3(Y1Y2Y3).
  DRAY_EXEC void get_vec_init_ptrs(int32 t_order, T *arr, T **ptrs)
  {
    // We align all the initial vectors in the same direction,
    // along the innermost index. (That is, as blocks of contiguous memory.)
    // Each vector above the 0th must clear the orders below it.
    ptrs[0] = arr;
    for (int32 idx = 1, offset = s; idx < t_order; idx++, offset*=s)
      ptrs[idx] = arr + offset;
  }

  // After storing data in the vectors (at addresses returned by get_vec_init_ptrs()),
  // use this to construct the tensor product.
  //
  // Returns: The size of the tensor constructed.
  DRAY_EXEC int32 construct_in_place(int32 t_order, T *arr)
  {
    // This is a recursive method.
    if (t_order == 0) { return s; }
    else
    {
      int32 size_below = construct_in_place(t_order - 1, arr);
      // The current vector is safe because it was stored out of the way of lower construct().
      // Now The first 'size_below' addresses hold the sub-product of the lower vectors.
      // Construct the final tensor by multiplying the sub-product by each component of the current vector.
      // To do this in place, must overwrite the sub-product AFTER using it for the rest of the tensor.
      const T *cur_vec = arr + size_below;
      const T cur_head = cur_vec[0];       // Save this ahead of time.
      for (int32 layer = s-1; layer >= 1; layer--)
      {
        const T comp = cur_vec[layer];
        memcpy(arr + layer * size_below, arr, size_below * sizeof(T));
        detail::MultInPlace<T>::mult(arr + layer * size_below, comp, size_below);
      }
      // Finish final layer by overwriting sub-product.
      detail::MultInPlace<T>::mult(arr, cur_head, size_below);
    }
  }
};


//// 
//// // --- Shape Type Public Interface --- //
//// //
//// // int32 get_el_dofs() const;
//// // int32 get_ref_dim() const;
//// // void calc_shape_dshape(const Array<int> &active_idx, const Array<Vec<RefDim>> &ref_pts, Array<T> &shape_val, Array<Vec<RefDim>> &shape_deriv) const; 
//// //
//// //
//// // --- Internal Parameters (example) --- //
//// //
//// // static constexpr int32 ref_dim;
//// // int32 p_order;
//// // int32 el_dofs;
//// 
//// // Abstract TensorShape class defines mechanics of tensor product.
//// // Inherit from this and define 2 methods:
//// // 1. Class method get_el_dofs_1d();
//// // 2. External function in the (template parameter) class Shape1D, calc_shape_dshape_1d();
//// template <typename T, int32 RefDim, typename Shape1D>
//// struct TensorShape
//// {
////   virtual int32 get_el_dofs_1d() const = 0;
//// 
////   int32 get_ref_dim() const { return RefDim; }
////   int32 get_el_dofs() const { return pow(get_el_dofs_1d(), RefDim); }
//// 
////   void calc_shape_dshape( const Array<int32> &active_idx,
////                           const Array<Vec<T,RefDim>> &ref_pts,
////                           Array<T> &shape_val,                      // Will be resized.
////                           Array<Vec<T,RefDim>> &shape_deriv) const;   // Will be resized.
//// };
//// 
//// 
//// template <typename T>
//// struct Bernstein1D
//// {
////   // Bernstein evaluator rippped out of MFEM.
////   DRAY_EXEC
////   static void calc_shape_dshape_1d(int32 el_dofs_1d, const T x, const T y, T *u, T *d)
////   {
////     const int32 p = el_dofs_1d - 1;
////     if (p == 0)
////     {
////        u[0] = 1.;
////        d[0] = 0.;
////     }
////     else
////     {
////        // Write binomial coefficients into u memory instead of allocating b[].
////        BinomRow<T>::fill_single_row(p,u);
//// 
////        const double xpy = x + y, ptx = p*x;
////        double z = 1.;
//// 
////        int i;
////        for (i = 1; i < p; i++)
////        {
////           //d[i] = b[i]*z*(i*xpy - ptx);
////           d[i] = u[i]*z*(i*xpy - ptx);
////           z *= x;
////           //u[i] = b[i]*z;
////           u[i] = u[i]*z;
////        }
////        d[p] = p*z;
////        u[p] = z*x;
////        z = 1.;
////        for (i--; i > 0; i--)
////        {
////           d[i] *= z;
////           z *= y;
////           u[i] *= z;
////        }
////        d[0] = -p*z;
////        u[0] = z*y;
////     }
////   }
//// 
////   DRAY_EXEC static bool IsInside(const T ref_coord)
////   {
////     //TODO some tolerance?  Where can we make watertight?
////     // e.g. Look at MFEM's Geometry::CheckPoint(geom, ip, ip_tol)
////     return 0.0 <= ref_coord  &&  ref_coord < 1.0;
////   }
//// };
//// 
//// template <typename T, int32 RefDim>
//// struct BernsteinShape : public TensorShape<T, RefDim, Bernstein1D<T>>
//// {
////   int32 m_p_order;
//// 
////   virtual int32 get_el_dofs_1d() const { return m_p_order + 1; }
//// 
////   // Inherits from TensorShape
////   // - int32 get_ref_dim() const;
////   // - int32 get_el_dofs() const;
////   // - void calc_shape_dshape(...) const;
//// 
////   DRAY_EXEC static bool IsInside(const Vec<T,RefDim> ref_pt)
////   {
////     for (int32 rdim = 0; rdim < RefDim; rdim++)
////     {
////       if (!Bernstein1D<T>::IsInside(ref_pt[rdim])) return false;
////     };
////     return true;
////   }
//// 
////   static BernsteinShape factory(int32 p)
////   {
////     BernsteinShape ret;
////     ret.m_p_order = p;
////     return ret;
////   }
//// };


//
// ShapeType Interface
//
// int32 get_el_dofs() const;
// int32 get_ref_dim() const;
//
//   // The number of auxiliary elements needed for linear_combo() parameter aux_mem.
// int32 get_size_aux() const;
// bool needs_aux_mem() const;
//
//   // Linear combination of value functions, and linear combinations of derivative functions.
//   // This is to evaluate a transformmation using a given set of control points at a given reference points.
// template <int32 PhysDim>
// DRAY_EXEC static void linear_combo(
//     const int32 p,
//     const Vec<T,RefDim> &xyz,
//     const Vec<T,PhysDim> *coeff,
//     Vec<T,PhysDim> &out_val,
//     Matrix<T,PhysDim,RefDim> &out_deriv,
//     T* aux_mem = NULL);
//
//   // If just want raw shape values/derivatives,
//   // stored in memory, to do something with them later:
// DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const; 
//
//
//   // TODO maybe some methods to change basis, and some methods to get (refined) bounds.


// TODO TODO TODO I was in the middle of this when I stopped to do SimpleTensor.
///template <typename T, int32 RefDim>
///struct BernsteinBasis
///{
///  int32 p;
///  BernsteinBasis(int32 _p) : p(_p) {}
///
///  DRAY_EXEC int32 get_el_dofs() const { return pow(p+1, RefDim); }
///  DRAY_EXEC int32 get_ref_dim() const { return RefDim; }
/// 
///    // The number of auxiliary elements needed for linear_combo() parameter aux_mem.
///  DRAY_EXEC int32 get_size_aux() const { return 2 * RefDim * (p+1); }
///  DRAY_EXEC bool needs_aux_mem() const { return true; }
/// 
///    // Linear combination of value functions, and linear combinations of derivative functions.
///    // This is to evaluate a transformmation using a given set of control points at a given reference points.
///  template <int32 PhysDim>
///  DRAY_EXEC static void linear_combo(
///      const int32 p,
///      const Vec<T,RefDim> &xyz,
///      const Vec<T,PhysDim> *coeff,
///      Vec<T,PhysDim> &out_val,
///      Matrix<T,PhysDim,RefDim> &out_deriv,
///      T* aux_mem = NULL);
/// 
///    // If just want raw shape values/derivatives,
///    // stored in memory, to do something with them later:
///  DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const; 
///
///protected:
///  BernsteinBasis() { assert(false); }
///
///};  // BernsteinBasis


//
// PowerBasis (ND)
//
template <typename T, int32 RefDim>
struct PowerBasis : public PowerBasis<T, RefDim-1>
{
  // -- Internals -- //

  int32 coeff_offset;  // Set by initialize().

    // Initializes p and coeff_offset, and returns offset.
  DRAY_EXEC int32 initialize(int32 p) { return coeff_offset = (p+1) * PowerBasis<T,RefDim-1>::initialize(p); }

  template <int32 PhysDim>
  DRAY_EXEC void m_linear_combo(const Vec<T,RefDim> &xyz, const Vec<T,PhysDim> *coeff,
      Vec<T,PhysDim> &ac_v, Vec<Vec<T,PhysDim>,RefDim> &ac_dxyz);


  // -- Public -- //

  int32 get_el_dofs() const { return (PowerBasis<T,1>::p + 1) * coeff_offset; }
  int32 get_ref_dim() const { return RefDim; }

  int32 get_size_aux() const { return 0; }
  bool needs_aux_mem() const { return false; }

  template <int32 PhysDim>
  DRAY_EXEC static void linear_combo(const int32 p, const Vec<T,RefDim> &xyz, const Vec<T,PhysDim> *coeff,
      Vec<T,PhysDim> &out_val, Matrix<T,PhysDim,RefDim> &out_deriv);

  // DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const;   //TODO

  // Non-existent option.
  template <int32 PhysDim>
  DRAY_EXEC static void linear_combo(const int32 p, const Vec<T,RefDim> &xyz, const Vec<T,PhysDim> *coeff,
      Vec<T,PhysDim> &out_val, Matrix<T,PhysDim,RefDim> &out_deriv, T *aux_mem) { assert(false); }

};  // PowerBasis ND


//
// PowerBasis (1D - specialization for base case)
//
template <typename T>
struct PowerBasis<T, 1>
{
  // -- Internals -- //
  int32 p;   // Used by higher dimensions.

    // Returns offset of 1.
  DRAY_EXEC int32 initialize(int32 _p) { p = _p; return 1; }

  template <int32 PhysDim>
  DRAY_EXEC void m_linear_combo( const Vec<T,1> &xyz, const Vec<T,PhysDim> *coeff, Vec<T,PhysDim> &ac_v, Vec<Vec<T,PhysDim>,1> &ac_dxyz)
  {
    PowerBasis<T,1>::linear_combo<PhysDim>(p, xyz[0], coeff, ac_v, ac_dxyz[0]);
  }


  // -- Public -- //

  int32 get_el_dofs() const { return p+1; }
  int32 get_ref_dim() const { return 1; }

  int32 get_size_aux() const { return 0; }
  bool needs_aux_mem() const { return false; }

  template <int32 PhysDim>
  DRAY_EXEC static void linear_combo( const int32 p, const T &x, const Vec<T,PhysDim> *coeff, Vec<T,PhysDim> &ac_v, Vec<T,PhysDim> &ac_dx)
  {
    ac_v = 0.0;
    ac_dx = 0.0;
    int32 k;
    for (k = p; k > 0; k--)
    {
      ac_v = ac_v * x + coeff[k];
      ac_dx = ac_dx * x + coeff[k] * k;
    }
    ac_v = ac_v * x + coeff[k];
  }

  // DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const;   //TODO

  // Non-existent option.
  template <int32 PhysDim>
  DRAY_EXEC static void linear_combo( const int32 p, const T &x, const Vec<T,PhysDim> *coeff,
      Vec<T,PhysDim> &ac_v, Vec<T,PhysDim> &ac_dx, T *aux_mem) { assert(false); }

};  // PowerBasis 1D


template <typename T, int32 RefDim>
template <int32 PhysDim>
DRAY_EXEC void
PowerBasis<T,RefDim>::m_linear_combo( const Vec<T,RefDim> &xyz, const Vec<T,PhysDim> *coeff, Vec<T,PhysDim> &ac_v, Vec<Vec<T,PhysDim>,RefDim> &ac_dxyz)
{
  // Local so compiler can figure it out.
  const int32 &p = PowerBasis<T,1>::p;

  // Initialize all accumulators to zero.
  ac_v = 0.0;
  for (int32 r=0; r<RefDim; r++)
    ac_dxyz[r] = 0.0;

  // Aliases to separate x-component from yz-components.
  const T &x                 = xyz[0];
  const Vec<T,RefDim-1> &yz  = *((const Vec<T,RefDim-1> *) &xyz[1]);

  Vec<T,PhysDim> &ac_dx                  = ac_dxyz[0];
  Vec<Vec<T,PhysDim>,RefDim-1> &ac_dyz  = *((Vec<Vec<T,PhysDim>,RefDim-1> *) &ac_dxyz[1]);

  // Variables to hold results of "inner" summations.
  Vec<T,PhysDim> ac_v_i;
  Vec<Vec<T,PhysDim>,RefDim-1> ac_dyz_i;  // Note dx is absent from inner.

  int32 k;
  for (k = p; k > 0; k--)
  {
    PowerBasis<T,RefDim-1>::m_linear_combo(
        yz, coeff + k * coeff_offset, ac_v_i, ac_dyz_i);
    ac_v = ac_v * x + ac_v_i;
    for (int32 r=0; r<RefDim-1; r++)
      ac_dyz[r] = ac_dyz[r] * x + ac_dyz_i[r];
    ac_dx = ac_dx * x + ac_v_i * k;
  }
  PowerBasis<T,RefDim-1>::m_linear_combo(
      yz, coeff + k * coeff_offset, ac_v_i, ac_dyz_i);
  ac_v = ac_v * x + ac_v_i;
  for (int32 r=0; r<RefDim-1; r++)
    ac_dyz[r] = ac_dyz[r] * x + ac_dyz_i[r];
}
///////DRAY_EXEC static void linear_combo_power_basis(
///////    const int32 p,
///////    const T x,
///////    const T y,
///////    const Vec<T,PhysDim> *coeff,
///////    Vec<T,PhysDim> &ac_v,
///////    Vec<T,PhysDim> &ac_dx,
///////    Vec<T,PhysDim> &ac_dy)
///////{
///////  ac_v = 0;
///////  ac_dx = 0;
///////  ac_dy = 0;
///////  Vec<T,PhysDim> ac_v_i;   // "inner"
///////  Vec<T,PhysDim> ac_dy_i;  // "inner"
///////  for (int32 k = p; k > 0; k--)
///////  {
///////    linear_combo_power_basis(p, y, coeff + k * (p+1), ac_v_i, ac_dy_i);
///////    ac_v = ac_v * x + ac_v_i;
///////    ac_dy = ac_dy * x + ac_dy_i;
///////    ac_dx = ac_dx * x + ac_v_i * k;
///////  }
///////  linear_combo_power_basis(p, y, coeff + k * (p+1), ac_v_i, ac_dy_i);
///////  ac_v = ac_v * x + ac_v_i;
///////  ac_dy = ac_dy * x + ac_dy_i;
///////}

template <typename T, int32 RefDim>
template <int32 PhysDim>
DRAY_EXEC void
PowerBasis<T,RefDim>::linear_combo(
    const int32 p, const Vec<T,RefDim> &xyz, const Vec<T,PhysDim> *coeff,
    Vec<T,PhysDim> &out_val, Matrix<T,PhysDim,RefDim> &out_deriv)
{
  PowerBasis pb;
  pb.initialize(p);

  Vec<Vec<T,PhysDim>,RefDim> result_deriv;

  pb.m_linear_combo<PhysDim>(xyz, coeff, out_val, result_deriv);

  for (int32 rdim = 0; rdim < RefDim; rdim++)
  {
    out_deriv.set_col(rdim, result_deriv[rdim]);
  }
}



// ElTrans

// ElTransQuery ????

// NewtonSolve



} // namespace dray

#endif



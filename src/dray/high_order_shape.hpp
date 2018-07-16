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
// ShapeOp Interface
//
//--//   template <typename T, int32 RefDim>
//--//   struct ShapeOp
//--//   {
//--//     static constexpr int32 ref_dim = RefDim;
//--//
//--//     int32 get_el_dofs() const;
//--//   
//--//     // Stateful operator, where state includes polynomial order, pointer to auxiliary memory, etc.
//--//     T *m_aux_mem_ptr;
//--//     void set_aux_mem_ptr(T *aux_mem_ptr) { m_aux_mem_ptr = aux_mem_ptr; }
//--//
//--//     // The number of auxiliary elements needed for linear_combo() parameter aux_mem.
//--//     int32 get_aux_req() const;
//--//     bool static is_aux_req();
//--//   
//--//     template <typename CoeffIterType>
//--//     DRAY_EXEC void linear_combo(const Vec<T,RefDim> &xyz,
//--//                                   const CoeffIterType &coeff_iter,
//--//                                   Vec<CoeffIterType::phys_dim> &result_val,
//--//                                   Vec<Vec<T,CoeffIterType::phys_dim>,RefDim> &result_deriv);
//--//
//--//     // If just want raw shape values/derivatives,
//--//     // stored in memory, to do something with them later:
//--//     DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const;   //Optional
//--//   };


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
// PowerBasis (Arbitrary dimension)
//
template <typename T, int32 RefDim>
struct PowerBasis : public PowerBasis<T, RefDim-1>
{
  // -- Internals -- //

  int32 m_coeff_offset;  // Set by init_shape().

  // -- Public -- //

    // Initializes p and coeff_offset, and returns offset.
  DRAY_EXEC int32 init_shape(int32 p) { return m_coeff_offset = (p+1) * PowerBasis<T,RefDim-1>::init_shape(p); }

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo(const Vec<T,RefDim> &xyz,
                                const CoeffIterType &coeff_iter,
                                Vec<T,PhysDim> &result_val,
                                Vec<Vec<T,PhysDim>,RefDim> &result_deriv) const;

  static constexpr int32 ref_dim = RefDim;
  int32 get_el_dofs() const { return (PowerBasis<T,1>::p + 1) * m_coeff_offset; }
  int32 get_ref_dim() const { return RefDim; }

  int32 get_aux_req() const { return 0; }
  bool is_aux_req() const { return false; }

  // DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const;   //TODO

};  // PowerBasis (Arbitrary dimension)


//
// PowerBasis (1D - specialization for base case)
//
template <typename T>
struct PowerBasis<T, 1>
{
  // -- Internals -- //
  int32 p;   // Used by higher dimensions.

  // -- Public -- //

    // Returns offset of 1.
  DRAY_EXEC int32 init_shape(int32 _p) { p = _p; return 1; }

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo( const Vec<T,1> &xyz, const CoeffIterType &coeff_iter, Vec<T,PhysDim> &ac_v, Vec<Vec<T,PhysDim>,1> &ac_dxyz) const
  {
    PowerBasis<T,1>::linear_combo<CoeffIterType,PhysDim>(p, xyz[0], coeff_iter, ac_v, ac_dxyz[0]);
  }

  static constexpr int32 ref_dim = 1;
  int32 get_el_dofs() const { return p+1; }

  int32 get_aux_req() const { return 0; }
  bool is_aux_req() const { return false; }

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC static void linear_combo( const int32 p, const T &x, const CoeffIterType &coeff_iter, Vec<T,PhysDim> &ac_v, Vec<T,PhysDim> &ac_dx)
  {
    ac_v = 0.0;
    ac_dx = 0.0;
    int32 k;
    for (k = p; k > 0; k--)
    {
      ac_v = ac_v * x + coeff_iter[k];
      ac_dx = ac_dx * x + coeff_iter[k] * k;
    }
    ac_v = ac_v * x + coeff_iter[k];
  }

  // DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const;   //TODO

};  // PowerBasis 1D

//
// PowerBasis<T,RefDim>::linear_combo()
//
template <typename T, int32 RefDim>
template <typename CoeffIterType, int32 PhysDim>
DRAY_EXEC void
PowerBasis<T,RefDim>::linear_combo( const Vec<T,RefDim> &xyz, const CoeffIterType &coeff_iter, Vec<T,PhysDim> &ac_v, Vec<Vec<T,PhysDim>,RefDim> &ac_dxyz) const
{
  // Local so compiler can figure it out.
  const int32 &p = PowerBasis<T,1>::p;

  // Local const so we don't modify ourself.
  const int32 coeff_offset = m_coeff_offset;

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
    PowerBasis<T,RefDim-1>::linear_combo(
        yz, coeff_iter + k * coeff_offset, ac_v_i, ac_dyz_i);
    ac_v = ac_v * x + ac_v_i;
    for (int32 r=0; r<RefDim-1; r++)
      ac_dyz[r] = ac_dyz[r] * x + ac_dyz_i[r];
    ac_dx = ac_dx * x + ac_v_i * k;
  }
  PowerBasis<T,RefDim-1>::linear_combo(
      yz, coeff_iter + k * coeff_offset, ac_v_i, ac_dyz_i);
  ac_v = ac_v * x + ac_v_i;
  for (int32 r=0; r<RefDim-1; r++)
    ac_dyz[r] = ac_dyz[r] * x + ac_dyz_i[r];
}
////// // The Idea.
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


//
//
// Notes
//
//

//---- Interface ----//
//
//--//   template <typename T, int32 RefDim>
//--//   struct ShapeOp
//--//   {
//--//     static constexpr int32 ref_dim = RefDim;
//--//   
//--//     // Stateful operator, where state includes polynomial order, pointer to auxiliary memory, etc.
//--//     T *m_aux_mem_ptr;
//--//     void set_aux_mem_ptr(T *aux_mem_ptr) { m_aux_mem_ptr = aux_mem_ptr; }
//--//
//--//     // The number of auxiliary elements needed for linear_combo() parameter aux_mem.
//--//     int32 get_aux_req() const;
//--//     bool static is_aux_req();
//--//   
//--//     template <typename CoeffIterType>
//--//     DRAY_EXEC void linear_combo(const Vec<T,RefDim> &xyz, const CoeffIterType &coeff_iter,
//--//                                   Vec<CoeffIterType::phys_dim> &result_val, Vec<Vec<T,CoeffIterType::phys_dim>,RefDim> &result_deriv);
//--//   };



template <typename T, int32 PhysDim>
struct ElTransIter
{
  static constexpr int32 phys_dim = PhysDim;   //TODO might have to define this in the implementation file as well.

  const int32 *m_el_dofs_ptr;        // Start of sub array, indexed by [dof_idx].
  const Vec<T,PhysDim> *m_val_ptr;  // Start of total array, indexed by m_el_dofs_ptr[dof_idx].

  int32 m_offset;

  DRAY_EXEC void init_iter(int32 *ctrl_idx_ptr, Vec<T,PhysDim> *val_ptr, int32 el_dofs, int32 el_id)
  {
    m_el_dofs_ptr = ctrl_idx_ptr + el_dofs * el_id;
    m_val_ptr = val_ptr;
    m_offset = 0;
  }
  
  DRAY_EXEC Vec<T,PhysDim> operator[] (int32 dof_idx)
  {
    dof_idx += m_offset;
    return m_val_ptr[m_el_dofs_ptr[dof_idx]];
  }

  DRAY_EXEC void operator+= (int32 dof_offset) { m_offset += dof_offset; }  // Less expensive
  DRAY_EXEC ElTransIter operator+ (int32 dof_offset);                       // More expensive
};

template <typename T, int32 PhysDim>
DRAY_EXEC ElTransIter<T,PhysDim>
ElTransIter<T,PhysDim>::operator+ (int32 dof_offset)
{
  ElTransIter<T,PhysDim> other = *this;
  other.m_offset += dof_offset;
  return other;
}


//
// ElTransBdryIter  -- To evaluate at only the boundary, using only boundary control points.
//                     Only for 3D Hex reference space, which has 6 2D faces as boundary.
//
template <typename T, int32 PhysDim>
struct ElTransBdryIter : public ElTransIter<T,PhysDim>
{
  using ElTransIter<T,PhysDim>::m_el_dofs_ptr;
  using ElTransIter<T,PhysDim>::m_val_ptr;
  using ElTransIter<T,PhysDim>::m_offset;

  // Members of this class.
  int32 m_el_dofs_1d;
  int32 m_stride_in, m_stride_out;

    // lowercase: 0_end. Uppercase: 1_end.
  enum class FaceID { x = 0, y = 1, z = 2, X = 3, Y = 4, Z = 5 };

  // There are 6 faces on a hex, so re-index the faces as new elements.
  // el_id_face = 6*el_id + face_id.
  DRAY_EXEC void init_iter(int32 *ctrl_idx_ptr, Vec<T,PhysDim> *val_ptr, int32 el_dofs_1d, int32 el_id_face)
  {
    int32 offset, stride_in, stride_out;
    const int32 d0 = 1;
    const int32 d1 = el_dofs_1d;
    const int32 d2 = d1 * el_dofs_1d;
    const int32 d3 = d2 * el_dofs_1d;
    switch (el_id_face % 6)
    {
      // Invariant: stride_out is a multiple of stride_in.
      case FaceID::x: offset = 0;       stride_in = d0; stride_out = d1; break;
      case FaceID::y: offset = 0;       stride_in = d0; stride_out = d2; break;
      case FaceID::z: offset = 0;       stride_in = d1; stride_out = d2; break;
      case FaceID::X: offset = d3 - d2; stride_in = d0; stride_out = d1; break;
      case FaceID::Y: offset = d2 - d1; stride_in = d0; stride_out = d2; break;
      case FaceID::Z: offset = d1 - d0; stride_in = d1; stride_out = d2; break;
    }

    m_el_dofs_1d = el_dofs_1d;
    m_stride_in = stride_in;
    m_stride_out = stride_out;
    m_el_dofs_ptr = ctrl_idx_ptr + d3 * (el_id_face / 6) + offset;
    m_val_ptr = val_ptr;
    m_offset = 0;
  }

  // 0 <= dof_idx < (el_dofs_1d)^2.
  DRAY_EXEC Vec<T,PhysDim> operator[] (int32 dof_idx)
  {
    dof_idx += m_offset;
    const int32 j = dof_idx % m_el_dofs_1d;
    const int32 i = dof_idx % (m_el_dofs_1d * m_el_dofs_1d) - j;
    return m_val_ptr[m_el_dofs_ptr[i*m_stride_out + j*m_stride_in]];
  }
};


//
// ElTransPairIter  -- To superimpose a vector field and scalar field over the same reference space,
//                     without necessarily having the same numbers of degrees of freedom.
//
template <typename T, int32 PhysDimX, int32 PhysDimY>
struct ElTransPairIter
{
  static constexpr int32 phys_dim = PhysDimX + PhysDimY;   //TODO might have to define this in the implementation file as well.

  const int32 *m_el_dofs_ptr_x;         // Start of sub array, indexed by [dof_idx].
  const int32 *m_el_dofs_ptr_y;         // Start of sub array, indexed by [dof_idx].
  const Vec<T,PhysDimX> *m_val_ptr_x;  // Start of total array, indexed by m_el_dofs_ptr_x[dof_idx].
  const Vec<T,PhysDimY> *m_val_ptr_y;  // Start of total array, indexed by m_el_dofs_ptr_y[dof_idx].

  int32 m_offset;

  DRAY_EXEC void init_iter(int32 *ctrl_idx_ptr_x, Vec<T,phys_dim> *val_ptr_x, int32 el_dofs_x,
                           int32 *ctrl_idx_ptr_y, Vec<T,phys_dim> *val_ptr_y, int32 el_dofs_y,
                           int32 el_id)
  {
    m_el_dofs_ptr_x = ctrl_idx_ptr_x + el_dofs_x * el_id;
    m_el_dofs_ptr_y = ctrl_idx_ptr_y + el_dofs_y * el_id;
    m_val_ptr_x = val_ptr_x;
    m_val_ptr_y = val_ptr_y;
    m_offset = 0;
  }
  
  DRAY_EXEC Vec<T,phys_dim> operator[] (int32 dof_idx)
  {
    dof_idx += m_offset;
    Vec<T,phys_dim> out;
    Vec<T,PhysDimX> &out_x = *((Vec<T,PhysDimX> *) &out);
    Vec<T,PhysDimY> &out_y = *((Vec<T,PhysDimY> *) &out[PhysDimX]);
    out_x = m_val_ptr_x[m_el_dofs_ptr_x[dof_idx]];
    out_x = m_val_ptr_y[m_el_dofs_ptr_y[dof_idx]];
    return out;
  }

  DRAY_EXEC void operator+= (int32 dof_offset) { m_offset += dof_offset; }   // Less expensive
  DRAY_EXEC ElTransPairIter operator+ (int32 dof_offset);                    // More expensive
};

template <typename T, int32 PhysDimX, int32 PhysDimY>
DRAY_EXEC ElTransPairIter<T,PhysDimX,PhysDimY>
ElTransPairIter<T,PhysDimX,PhysDimY>::operator+ (int32 dof_offset)
{
  ElTransPairIter<T,PhysDimX,PhysDimY> other = *this;
  other.m_offset += dof_offset;
  return other;
}


template <typename T, class ShapeOpType, typename CoeffIterType>
struct ElTransOp : public ShapeOpType
{
  static constexpr int32 phys_dim = CoeffIterType::phys_dim;
  static constexpr int32 ref_dim = ShapeOpType::ref_dim;

  CoeffIterType m_coeff_iter;

  DRAY_EXEC void eval(const Vec<T,ref_dim> &ref, Vec<T,phys_dim> &result_val,
                      Vec<Vec<T,phys_dim>,ref_dim> &result_deriv)
  {
    ShapeOpType::linear_combo(ref, m_coeff_iter, result_val, result_deriv);
  }
};

template <typename T, int32 PhysDim>
struct ElTransData
{
  Array<int32> m_ctrl_idx;    // 0 <= ii < size_el, 0 <= jj < el_dofs, 0 <= m_ctrl_idx[ii*el_dofs + jj] < size_ctrl
  Array<Vec<T,PhysDim>> m_values;   // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

  int32 m_el_dofs;
  int32 m_size_el;
  int32 m_size_ctrl;

  void resize(int32 size_el, int32 el_dofs, int32 size_ctrl);
};

//
// ElTransRayOp - Special purpose combination of element transformation and rays,
//                  PHI(u,v,...) - r(s),
//                where u,v,... are parametric space coordinates,
//                and s is distance along the ray.
//
//                Required: RayPhysDim <= ElTransOpType::phys_dim.
//
template <typename T, class ElTransOpType, int32 RayPhysDim>
struct ElTransRayOp : public ElTransOpType
{
  static constexpr int32 ref_dim = ElTransOpType::ref_dim + 1;
  static constexpr int32 phys_dim = ElTransOpType::phys_dim;

  Vec<T,phys_dim> m_minus_ray_dir;

  DRAY_EXEC void set_minus_ray_dir(const Vec<T,phys_dim> &ray_dir) { m_minus_ray_dir = -ray_dir; }

  // Override eval().
  DRAY_EXEC void eval(const Vec<T,ref_dim> &uvws, Vec<T,phys_dim> &result_val,
                      Vec<Vec<T,phys_dim>,ref_dim> &result_deriv)
  {
    // Decompose uvws into disjoint reference coordinates.
    constexpr int32 uvw_dim = ElTransOpType::ref_dim;
    const Vec<T,uvw_dim> &uvw = *((const Vec<T,uvw_dim> *) &uvws);
    const T &s = *((const T *) &uvws[uvw_dim]);

    // Sub array of derivatives corresponding to uvw reference dimensions.
    Vec<Vec<T,phys_dim>,uvw_dim> &uvw_deriv = *((Vec<Vec<T,phys_dim>,uvw_dim> *) &result_deriv);

    ElTransOpType::eval(uvw, result_val, uvw_deriv);

    for (int32 pdim = 0; pdim < RayPhysDim; pdim++)
    {
      result_val[pdim] += m_minus_ray_dir[pdim] * s;
    }
    result_deriv[uvw_dim] = m_minus_ray_dir;
  }
};


template <typename T>
struct NewtonSolve
{
  enum SolveStatus
  {
    NotConverged = 0,
    ConvergePhys = 1,
    ConvergeRef = 2
  };

  // solve() - The element id is implicit in trans.m_coeff_iter.
  //           The "initial guess" ref pt is set by the caller in [in]/[out] param "ref".
  //           The returned solution ref pt is set by the function in [in]/[out] "ref".
  //
  template <class TransOpType>
  DRAY_EXEC static SolveStatus solve(
      TransOpType &trans,
      const Vec<T,TransOpType::phys_dim> &target, Vec<T,TransOpType::ref_dim> &ref,
      const T tol_phys, const T tol_ref,
      int32 &steps_taken, const int32 max_steps = 10);
};


template <typename T>
  template <class TransOpType>
DRAY_EXEC typename NewtonSolve<T>::SolveStatus
NewtonSolve<T>::solve(
    TransOpType &trans,
    const Vec<T,TransOpType::phys_dim> &target,
    Vec<T,TransOpType::ref_dim> &ref,
    const T tol_phys,
    const T tol_ref,
    int32 &steps_taken,
    const int32 max_steps)
{
  // The element id is implicit in trans.m_coeff_iter.
  // The "initial guess" reference point is set in the [in]/[out] argument "ref".

  constexpr int32 phys_dim = TransOpType::phys_dim;
  constexpr int32 ref_dim = TransOpType::ref_dim;
  assert(phys_dim == ref_dim);   // Need square jacobian.

  Vec<T,ref_dim>                x = ref;
  Vec<T,phys_dim>               y, delta_y;
  Vec<Vec<T,phys_dim>,ref_dim>  deriv_cols;

  NewtonSolve<T>::SolveStatus convergence_status;  // return value.

  // Evaluate at current ref pt and measure physical error.
  trans.eval(x, y, deriv_cols);
  delta_y = target - y;
  convergence_status = (delta_y.norm < tol_phys) ? ConvergePhys : NotConverged;

  steps_taken = 0;
  while (steps_taken < max_steps && convergence_status == NotConverged)
  {
    // Store the derivative columns in matrix format.
    Matrix<T,phys_dim,ref_dim> jacobian;
    for (int32 rdim = 0; rdim < ref_dim; rdim++)
    {
      jacobian.set_col(rdim, deriv_cols[rdim]);
    }

    // Compute delta_x by hitting delta_y with the inverse of jacobian.
    bool inverse_valid;
    Vec<T,ref_dim> delta_x;
    delta_x = matrix_mult_inv(jacobian, delta_y, inverse_valid);  //Compiler error if ref_dim != phys_dim.

    if (inverse_valid)
    {
      // Apply the Newton increment.
      x = x + delta_x;
      steps_taken++;

      // If converged, we're done.
      convergence_status = (delta_x.norm < tol_ref) ? ConvergeRef : NotConverged;
      if (convergence_status == ConvergeRef)
        break;
    }
    else
    {
      // Uh-oh. Some kind of singularity.
      break;
    }

    // Evaluate at current ref pt and measure physical error.
    trans.eval(x, y, deriv_cols);
    delta_y = target - y;
    convergence_status = (delta_y.norm < tol_phys) ? ConvergePhys : NotConverged;
  }  // end while

  ref = x;
  return convergence_status;
}



} // namespace dray

#endif



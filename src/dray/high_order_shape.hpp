#ifndef DRAY_HIGH_ORDER_SHAPE_HPP
#define DRAY_HIGH_ORDER_SHAPE_HPP

#include <dray/array.hpp>
#include <dray/matrix.hpp>
#include <dray/vec.hpp>
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
// void calc_shape_dshape(const Array<int> &active_idx, const Array<Vec<RefDim>> &ref_pts, Array<T> &shape_val, Array<Vec<RefDim>> &shape_deriv) const; 
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
                          const Array<Vec<T,RefDim>> &ref_pts,
                          Array<T> &shape_val,                      // Will be resized.
                          Array<Vec<T,RefDim>> &shape_deriv) const;   // Will be resized.
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
template <typename T, int32 _PhysDim, int32 _RefDim, typename _ShapeType>
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
public:
  static constexpr int32 PhysDim = _PhysDim;
  static constexpr int32 RefDim = _RefDim;
  using ShapeType = _ShapeType;

private:
  int32 m_el_dofs;
  int32 m_size_el;
  int32 m_size_ctrl;
  ShapeType m_shape;
  Array<int32> m_ctrl_idx;    // 0 <= ii < size_el, 0 <= jj < el_dofs, 0 <= m_ctrl_idx[ii*el_dofs + jj] < size_ctrl
  Array<Vec<T,PhysDim>> m_values;   // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

public:
  void resize(int32 size_el, int32 el_dofs, ShapeType shape, int32 size_ctrl);

  // This method assumes that output arrays are already the right size.
  // It does not resize or assign new arrays to output parameters.
  void eval(const Array<int> &active_idx,
            const Array<int32> &el_ids, const Array<Vec<T,_RefDim>> &ref_pts,
            Array<Vec<T,_PhysDim>> &trans_val, Array<Matrix<T,_PhysDim,_RefDim>> &trans_deriv) const;

  // Clients may read and write contents of member arrays, but not size of member arrays.
  ArrayFS<int32>                get_m_ctrl_idx()              { ArrayFS<int32> a; a.set(m_ctrl_idx); return a; }
  ArrayFS<Vec<T,PhysDim>>       get_m_values()                { ArrayFS<Vec<T,PhysDim>> a; a.set(m_values); return a; }
  const ArrayFS<int32>          get_m_ctrl_idx_const() const  { ArrayFS<int32> a; a.set_const(m_ctrl_idx); return a; }
  const ArrayFS<Vec<T,PhysDim>> get_m_values_const()   const  { ArrayFS<Vec<T,PhysDim>> a; a.set_const(m_values); return a; }
};

template <typename T, int32 PhysDim, int32 RefDim>
using ElTrans_BernsteinShape = ElTrans<T,PhysDim,RefDim, BernsteinShape<T,RefDim>>;

// -----------

///template <int32 S>
///struct SizeSingle
///{
///  constexpr int32 size = S;
///  const 
///};
///
///template <typename SL, typename SR>
///struct SizePair
///{
///  constexpr int32 size = SL::size + SR::size;
///  typedef SL sl;
///  typedef SR sr;
///};


//TODO remove
template <int32 S>
struct PtrBundle
{
  void * ptrs[S];  // These are void* because we may mix pointers to different sized Vec.
  int32 widths[S];
};

template <int32 S>
struct PtrBundleConst
{
  const void * ptrs[S];
  int32 widths[S];
};


// Iterative sum.
template <typename RetValT, int32 NumQ>
struct QuerySum
{
  RetValT *m_ptrs[NumQ];

  DRAY_EXEC
  void get(int32 idx, RetValT &ret)
  {
    ret = 0;
    for (int32 arr = 0; arr < NumQ; arr++)
    {
      ret += m_ptrs[arr][idx];
    }
  }
};

// Template-binary tree concatenation.
template <typename RetValT, typename QueryCatA, typename QueryCatB>
struct QueryCat
{
  static constexpr int32 ref_dim = QueryCatA::ref_dim + QueryCatB::ref_dim;
  static constexpr int32 num_q = QueryCatA::num_q + QueryCatB::num_q;

  void * m_ptrs[num_q];  // QueryCatBase<ArrayCompT> knows what its pointer really is.

  DRAY_EXEC
  void set(int32 idx, const RetValT &val)
  {
    //TODO
  }

  template <typename RetValTT>   // May be returning into a higher compound.
  DRAY_EXEC
  static void get(const void **ptrs, int32 idx, RetValTT &ret, int32 ret_offset)
  {
    QueryCatA::get(ptrs,                    idx, ret, 0);
    QueryCatB::get(ptrs + QueryCatA::num_q, idx, ret, QueryCatA::ref_dim);
  }

  template <typename SetValTT>
  DRAY_EXEC
  static void set(void **ptrs, int32 idx, const SetValTT &val, int32 s, int32 e)
  {
    //TODO
  }

  DRAY_EXEC
  void get(int32 idx, RetValT &ret) const
  {
    QueryCat::get<RetValT>((const void **) m_ptrs, idx, ret, 0);
  }

};

namespace detail
{
// Cheat to use Vec::operator[]() or Matrix::get/set_col().
template <typename DestT, typename SrcT>
struct ColumnCopy
{
  DRAY_EXEC
  static void cpy(DestT &dest, int32 d_idx, const SrcT &src, int32 s_idx);
};

// Vec <- Vec
template <typename T, int32 DestS, int32 SrcS>
struct ColumnCopy<Vec<T,DestS>, Vec<T,SrcS>>
{
  DRAY_EXEC
  static void cpy(Vec<T,DestS> &dest, int32 d_idx, const Vec<T,SrcS> &src, int32 s_idx)
  {
    dest[d_idx] = src[s_idx];
  }
};

// Matrix <- Matrix
template <typename T, int32 DestR, int32 DestC, int32 SrcR, int32 SrcC>
struct ColumnCopy<Matrix<T,DestR,DestC>, Matrix<T,SrcR,SrcC>>
{
  DRAY_EXEC
  static void cpy(Matrix<T,DestR,DestC> &dest, int32 d_idx, const Matrix<T,SrcR,SrcC> &src, int32 s_idx)
  {
    dest.set_col(d_idx, src.get_col(s_idx));
  }
};

}   // namespace detail.


// Base case of the template-binary tree concatenation.
template <int32 RefDim, typename ArrayCompT>
struct QueryCatBase
{
  static constexpr int32 ref_dim = RefDim;
  static constexpr int32 num_q = 1;

  ArrayCompT *m_ptr;

  DRAY_EXEC
  void get(int32 idx, ArrayCompT &ret) const
  {
    ret = m_ptr[idx];
  }

  template <typename RetValTT>
  DRAY_EXEC
  static void get(const void **ptrs, int32 idx, RetValTT &ret, int32 ret_offset)
  {
    // Single pointer to single array. num_q == 1. Get val in array at idx.
    const ArrayCompT val = ((const ArrayCompT *) *ptrs)[idx];

    // Over ref_dims of array component type.
    for (int32 in_dim = 0, out_dim = ret_offset; in_dim < ref_dim; in_dim++, out_dim++)
    {
      detail::ColumnCopy<RetValTT,ArrayCompT>::cpy(ret, out_dim, val, in_dim);
      //ret.set_col(out_dim, val.get_col(in_dim));
    }
  }
};

template <typename T, int32 PhysDim, int32 RefDim, int32 NumQ>
class ElTransCmpdQuery
{
public:
  static constexpr int32 phys_dim = PhysDim;
  static constexpr int32 ref_dim = RefDim;
  static constexpr int32 num_q = NumQ;

////  // Derived classes are responsible for populating a PtrBundle
////  // with pointers to the appropriate arrays.
////    // For now, assume device pointers. TODO template on an enum or something
////  virtual void perform(const Array<int32> &active_idx) = 0;
////  virtual PtrBundleConst<NumQ> get_val_ptrs_const() const = 0;
////  virtual PtrBundleConst<NumQ> get_deriv_ptrs_const() const = 0;
////  virtual PtrBundleConst<NumQ> get_ref_pt_ptrs_const() const = 0;
////  virtual PtrBundle<NumQ> get_ref_pt_ptrs() = 0;

  //////  DRAY_EXEC
  //////  static get_val(PtrBundleConst<NumQ> bundle_val, int32 q_idx, Vec<T,PhysDim> &val)
  //////  {
  //////    val = static_cast<T>(0.0f);
  //////    for (int32 ii = 0; ii < NumQ; ii++)
  //////      val += ((Vec<T,PhysDim> *) bundle_val.ptrs[ii]) [q_idx];
  //////    // widths is not used.
  //////  }

  //////  DRAY_EXEC
  //////  static get_deriv(PtrBundleConst<NumQ> bundle_deriv, int32 q_idx, Matrix<T,PhysDim,RefDim> &deriv)
  //////  {
  //////    for (int32 ii = 0; ii < NumQ; ii++)
  //////    {
  //////      // Get column.  TODO    here we have a problem because the matrix pointer is void*,
  //////      // but we need it to know how many rows and columns it has, so that we can ask it get_col(jj);
  //////      // So, I'll look into getting a compile-time 'list' of widths using variadic template parameters?
  //////      deriv
  //////    }
  //////  }

  ////// DRAY_EXEC
  ////// static get_ref(PtrBundleConst<NumQ> bundle_ref, int32 q_idx, Vec<T,RefDim> &ref_pt);

  ////// DRAY_EXEC
  ////// static set_ref(PtrBundle<NumQ> bundle_ref, int32 q_idx, const Vec<T, RefDim> &ref_pt);
};

/*
 * An ElTransQuery returns a value/derivative that may be part of a larger query.
 *
 * The methods which actually return values must do so per array column, across a struct of arrays.
 * Therefore they must be DRAY_EXEC and 
 */




template <typename T, typename ElTransType>
///class ElTransQuery : public ElTransCmpdQuery<T, 1>
class ElTransQuery
{
  static constexpr int32 phys_dim = ElTransType::PhysDim;
  static constexpr int32 ref_dim = ElTransType::RefDim;
  static constexpr int32 num_q = 1;  //self, this is leaf.

  Array<int32> m_elt_id;
  Array<Vec<T,ref_dim>> m_ref_pts;
  Array<Vec<T,phys_dim>> m_result_val;
  Array<Matrix<T,phys_dim,ref_dim>> m_result_deriv;

  void perform(const Array<int32> &active_idx);
  PtrBundleConst<num_q> get_val_ptrs_const() const;
  PtrBundleConst<num_q> get_deriv_ptrs_const() const;
  PtrBundleConst<num_q> get_ref_pt_ptrs_const() const;
  PtrBundle<num_q> get_ref_pt_ptrs() = 0;

  DRAY_EXEC
  static void get_val(void * bundle_val[], int32 q_idx, Vec<T,phys_dim> &val)
  {
    val = ((Vec<T,phys_dim>*) bundle_val[0]) [q_idx];
  }
  DRAY_EXEC
  static Vec<T,phys_dim> get_val(void ** bundle_val, int32 q_idx)
  {
    return ((Vec<T,phys_dim>*) bundle_val[0]) [q_idx];
  }

  DRAY_EXEC
  static void get_deriv(void * bundle_deriv[], int32 q_idx, Matrix<T,phys_dim,ref_dim> &deriv)
  {
    //TODO
  }

  DRAY_EXEC
  static void get_ref(void * bundle_ref[], int32 q_idx, Vec<T,ref_dim> &ref_pt)
  {
    //TODO
  }

  DRAY_EXEC
  static void set_ref(void *bundle_ref[], int32 q_idx, const Vec<T, ref_dim> &ref_pt)
  {
    //TODO
  }
};

// Pair (binary tree) of ElTransQueries
template <typename T, typename QTypeA, typename QTypeB>
//////class ElTransQueryPair : public ElTransCmpdQuery<T, QTypeA::num_q, QTypeB::num_q>
struct ElTransQueryPair
{
  static constexpr int32 phys_dim = QTypeA::phys_dim;
  static constexpr int32 ref_dim = QTypeA::ref_dim + QTypeB::ref_dim;
  static constexpr int32 num_q = QTypeA::num_q + QTypeB::num_q;

  QTypeA query_a;
  QTypeB query_b;

  void perform(const Array<int32> &active_idx);
  PtrBundleConst<num_q> get_val_ptrs_const() const;
  PtrBundleConst<num_q> get_deriv_ptrs_const() const;
  PtrBundleConst<num_q> get_ref_pt_ptrs_const() const;
  PtrBundle<num_q> get_ref_pt_ptrs() = 0;

  DRAY_EXEC
  static void get_val(void * bundle_val[], int32 q_idx, Vec<T,phys_dim> &val)
  {
    val = QTypeA::get_val((void**) bundle_val) +
          QTypeB::get_val((void**) bundle_val + QTypeA::num_q);
  }
  DRAY_EXEC
  static Vec<T,phys_dim> get_val(void ** bundle_val, int32 q_idx)
  {
    return QTypeA::get_val(bundle_val) +
           QTypeB::get_val(bundle_val + QTypeA::num_q);
  }

  DRAY_EXEC
  static void get_deriv(void * bundle_deriv[], int32 q_idx, Matrix<T,phys_dim,ref_dim> &deriv)
  {
    //TODO
  }

  DRAY_EXEC
  static void get_ref(void * bundle_ref[], int32 q_idx, Vec<T,ref_dim> &ref_pt)
  {
    //TODO
  }

  DRAY_EXEC
  static void set_ref(void * bundle_ref[], int32 q_idx, const Vec<T, ref_dim> &ref_pt)
  {
    //TODO
  }
};


} // namespace dray

#endif

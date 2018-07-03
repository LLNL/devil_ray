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
template <typename _T, int32 _PhysDim, int32 _RefDim, typename _ShapeType>
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
  using T = _T;
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

template <int32 S>
struct PtrBundle
{
  void * ptrs[S];  // These are void* because we may mix pointers to different sized Vec.
};


template <typename T>
struct QSum
{
  DRAY_EXEC static T get(const PtrBundle<2> &ptrb, const int32 &idx)
  {
    return ((T*) ptrb.ptrs[0])[idx] + ((T*) ptrb.ptrs[1])[idx];
  };
};

template <typename T3, typename T1, typename T2>
struct QCat
{
  DRAY_EXEC static T3 get(const PtrBundle<2> &ptrb, const int32 &idx);
  DRAY_EXEC static void set(PtrBundle<2> &ptrb, const int32 &idx, const T3 &val);
};

// Specializations for Vec cat and Matrix cat.
template <typename T, int32 S1, int32 S2>
struct QCat<Vec<T,S1+S2>, Vec<T,S1>, Vec<T,S2>>
{
  DRAY_EXEC static Vec<T,S1+S2> get(const PtrBundle<2> &ptrb, const int32 &idx)
  {
    Vec<T,S1+S2> ret;
    int32 ii;
    for (ii = 0; ii < S1; ii++)
      ret[ii] = ((Vec<T,S1> *) ptrb.ptrs[0])[idx][ii];
    for (int32 jj = 0; jj < S2; jj++, ii++)
      ret[ii] = ((Vec<T,S2> *) ptrb.ptrs[0])[idx][jj];
    return ret;
  }

  DRAY_EXEC static void set(PtrBundle<2> &ptrb, const int32 &idx, const Vec<T,S1+S2> &val)
  {
    int32 ii;
    for (ii = 0; ii < S1; ii++)
      ((Vec<T,S1> *) ptrb.ptrs[0])[idx][ii] = val[ii];
    for (int32 jj = 0; jj < S2; jj++, ii++)
      ((Vec<T,S2> *) ptrb.ptrs[0])[idx][jj] = val[ii];
  }
};

template <typename T, int32 R, int32 C1, int32 C2>
struct QCat<Matrix<T,R,C1+C2>, Matrix<T,R,C1>, Matrix<T,R,C2>>
{
  DRAY_EXEC static Matrix<T,R,C1+C2> get(const PtrBundle<2> &ptrb, const int32 &idx)
  {
    Matrix<T,R,C1+C2> ret;
    int32 ii;
    for (ii = 0; ii < C1; ii++)
      ret.set_col(ii, ((Matrix<T,R,C1> *) ptrb.ptrs[0])[idx].get_col(ii) );
    for (int32 jj = 0; jj < C2; jj++, ii++)
      ret.set_col(ii, ((Matrix<T,R,C2> *) ptrb.ptrs[0])[idx].get_col(jj) );
    return ret;
  }

  DRAY_EXEC static void set(const PtrBundle<2> &ptrb, const int32 &idx, const Matrix<T,R,C1+C2> &val)
  {
    int32 ii;
    for (ii = 0; ii < C1; ii++)
      ((Matrix<T,R,C1> *) ptrb.ptrs[0])[idx].set_col(ii, val.get_col(ii) );
    for (int32 jj = 0; jj < C2; jj++, ii++)
      ((Matrix<T,R,C2> *) ptrb.ptrs[0])[idx].set_col(jj, val.get_col(ii) );
  }
};


//
// ElTransQuery (single field)
//
template <typename ElTransType>
struct ElTransQuery
{
  static constexpr int32 num_q = 1;
  typedef PtrBundle<num_q> ptr_bundle_t;

  using T = typename ElTransType::T;
  static constexpr int32 phys_dim = ElTransType::PhysDim;
  static constexpr int32 ref_dim = ElTransType::RefDim;

  Array<int32> m_el_ids;
  Array<Vec<T,ref_dim>> m_ref_pts;
  Array<Vec<T,phys_dim>> m_result_val;
  Array<Matrix<T,phys_dim,ref_dim>> m_result_deriv;

  void resize(const size_t size)
  {
    m_el_ids.resize(size);
    m_ref_pts.resize(size);
    m_result_val.resize(size);
    m_result_deriv.resize(size);
  }

  // Query an ElTrans.
  void query(const ElTransType &eltrans, const Array<int32> &active_idx)
  {
    eltrans.eval(active_idx, m_el_ids, m_ref_pts, m_result_val, m_result_deriv);
  }

  // Device pointers.
  const ptr_bundle_t get_val_device_ptr_const() const
  {
    return {(void*) m_result_val.get_device_ptr_const()};
  }

  const ptr_bundle_t get_deriv_device_ptr_const() const
  {
    return {(void*) m_result_deriv.get_device_ptr_const()};
  } 

  const ptr_bundle_t get_ref_device_ptr_const() const
  {
    return {(void*) m_ref_pts.get_device_ptr_const()};
  }
 
  ptr_bundle_t get_ref_device_ptr()
  {
    return {(void*) m_ref_pts.get_device_ptr()};
  }

  // Array element access.
  DRAY_EXEC static Vec<T,phys_dim> get_val(const ptr_bundle_t &ptrb, int32 idx)
  {
    return ((Vec<T,phys_dim> *) ptrb.ptrs[0])[idx];
  }

  DRAY_EXEC static Matrix<T,phys_dim,ref_dim> get_deriv(const ptr_bundle_t &ptrb, int32 idx)
  {
    return ((Matrix<T,phys_dim,ref_dim> *) ptrb.ptrs[0])[idx];
  }

  DRAY_EXEC static Vec<T,phys_dim> get_ref(const ptr_bundle_t &ptrb, int32 idx)
  {
    return ((Vec<T,ref_dim> *) ptrb.ptrs[0])[idx];
  }

  DRAY_EXEC static void set_ref(ptr_bundle_t &ptrb, int32 idx, const Vec<T,ref_dim> &ref)
  {
    ((Vec<T,ref_dim> *) ptrb.ptrs[0])[idx] = ref;
  }
};  // ElTransQuery



//
// ElTransQuery2 (two fields over disjoint reference spaces)
//
template <typename ElTransType1, typename ElTransType2>
struct ElTransQuery2
{
  static constexpr int32 num_q = 2;
  typedef PtrBundle<num_q> ptr_bundle_t;

  using T = typename ElTransType1::T;
  static constexpr int32 phys_dim = ElTransType1::PhysDim;
  static constexpr int32 ref_dim1 = ElTransType1::RefDim;
  static constexpr int32 ref_dim2 = ElTransType2::RefDim;
  static constexpr int32 ref_dim = ref_dim1 + ref_dim2;

  ElTransQuery<ElTransType1> m_q1;
  ElTransQuery<ElTransType2> m_q2;

  void resize(const size_t size)
  {
    m_q1.resize(size);
    m_q2.resize(size);
  }

  // Query two ElTrans objects representing two disjoint fields.
  void query(const ElTransType1 &eltrans1, const ElTransType2 &eltrans2, const Array<int32> &active_idx)
  {
    m_q1.query(eltrans1, active_idx);
    m_q2.query(eltrans2, active_idx);
  }

  // Device pointers.
  const ptr_bundle_t get_val_device_ptr_const() const
  {
    return {(void*) m_q1.m_result_val.get_device_ptr_const(),
            (void*) m_q2.m_result_val.get_device_ptr_const() };
  }

  const ptr_bundle_t get_deriv_device_ptr_const() const
  {
    return {(void*) m_q1.m_result_deriv.get_device_ptr_const(),
            (void*) m_q2.m_result_deriv.get_device_ptr_const() };
  } 

  const ptr_bundle_t get_ref_device_ptr_const() const
  {
    return {(void*) m_q1.m_ref_pts.get_device_ptr_const(),
            (void*) m_q2.m_ref_pts.get_device_ptr_const() };
  }
 
  ptr_bundle_t get_ref_device_ptr()
  {
    return {(void*) m_q1.m_ref_pts.get_device_ptr(),
            (void*) m_q2.m_ref_pts.get_device_ptr() };
  }

  // Array element access.
  DRAY_EXEC static Vec<T,phys_dim> get_val(const ptr_bundle_t &ptrb, int32 idx)
  {
    return QSum<Vec<T,phys_dim>>::get(ptrb, idx);
  }

  DRAY_EXEC static Matrix<T,phys_dim,ref_dim> get_deriv(const ptr_bundle_t &ptrb, int32 idx)
  {
    return QCat<Matrix<T,phys_dim,ref_dim>,
                Matrix<T,phys_dim,ref_dim1>,
                Matrix<T,phys_dim,ref_dim2>>::get(ptrb, idx);
  }

  DRAY_EXEC static Vec<T,phys_dim> get_ref(const ptr_bundle_t &ptrb, int32 idx)
  {
    return QCat<Vec<T,ref_dim>,
                Vec<T,ref_dim1>,
                Vec<T,ref_dim2>>::get(ptrb, idx);
  }

  DRAY_EXEC static void set_ref(ptr_bundle_t &ptrb, int32 idx, const Vec<T,ref_dim> &ref)
  {
    return QCat<Vec<T,ref_dim>,
                Vec<T,ref_dim1>,
                Vec<T,ref_dim2>>::set(ptrb, idx, ref);
  }
};  // ElTransQuery2

} // namespace dray

#endif



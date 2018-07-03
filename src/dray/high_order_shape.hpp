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

////template <int32 S>
////struct PtrBundleConst
////{
////  const void * ptrs[S];  // These are void* because we may mix pointers to different sized Vec.
////};

template <typename T>
struct QSum : public PtrBundle<2>
{
  DRAY_EXEC
  const T& operator[] (const int32 &idx) const
  {
    return ((T*) ptrs[0])[idx] + ((T*) ptrs[1])[idx];
  };
};

template <typename T3, typename T1, typename T2>
struct QCat : public PtrBundle<2>
{
  DRAY_EXEC
  const T3& operator[] (const int32 &idx) const
  {
    //TODO
  }

  DRAY_EXEC
  T3& operator[] (const int32 &idx)
  {
    //TODO
  }
};


//template <typename T, int32 PhysDim, int32 RefDim>
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
  DRAY_EXEC
  static Vec<T,phys_dim> get_val(const ptr_bundle_t &ptrb, int32 idx)
  {
    return ((Vec<T,phys_dim> *) ptrb.ptrs[0])[idx];
  }

  DRAY_EXEC
  static Matrix<T,phys_dim,ref_dim> get_deriv(const ptr_bundle_t &ptrb, int32 idx)
  {
    return ((Matrix<T,phys_dim,ref_dim> *) ptrb.ptrs[0])[idx];
  }

  DRAY_EXEC
  static Vec<T,phys_dim> get_ref(const ptr_bundle_t &ptrb, int32 idx)
  {
    return ((Vec<T,ref_dim> *) ptrb.ptrs[0])[idx];
  }

  DRAY_EXEC
  static void set_ref(ptr_bundle_t &ptrb, int32 idx, const Vec<T,ref_dim> &ref)
  {
    ((Vec<T,ref_dim> *) ptrb.ptrs[0])[idx] = ref;
  }
};





} // namespace dray

#endif






/////////
// THE ABYSS
////////////
//



//////  // Iterative sum.
//////  template <typename RetValT, int32 NumQ>
//////  struct QuerySum
//////  {
//////    RetValT *m_ptrs[NumQ];
//////  
//////    DRAY_EXEC
//////    void get(int32 idx, RetValT &ret)
//////    {
//////      ret = 0;
//////      for (int32 arr = 0; arr < NumQ; arr++)
//////      {
//////        ret += m_ptrs[arr][idx];
//////      }
//////    }
//////  };
//////  
//////  //TODO look at variadic templates, see if we can use them to make the following iterative also.
//////  
//////  // Template-binary tree concatenation.
//////  template <typename RetValT, typename QueryCatA, typename QueryCatB>
//////  struct QueryCat
//////  {
//////    static constexpr int32 ref_dim = QueryCatA::ref_dim + QueryCatB::ref_dim;
//////    static constexpr int32 num_q = QueryCatA::num_q + QueryCatB::num_q;
//////  
//////    void * m_ptrs[num_q];  // QueryCatBase<ArrayCompT> knows what its pointer really is.
//////  
//////    DRAY_EXEC
//////    void set(int32 idx, const RetValT &val)
//////    {
//////      //TODO
//////    }
//////  
//////    template <typename RetValTT>   // May be returning into a higher compound.
//////    DRAY_EXEC
//////    static void get(const void **ptrs, int32 idx, RetValTT &ret, int32 ret_offset)
//////    {
//////      QueryCatA::get(ptrs,                    idx, ret, 0);
//////      QueryCatB::get(ptrs + QueryCatA::num_q, idx, ret, QueryCatA::ref_dim);
//////    }
//////  
//////    template <typename SetValTT>
//////    DRAY_EXEC
//////    static void set(void **ptrs, int32 idx, const SetValTT &val, int32 s, int32 e)
//////    {
//////      //TODO
//////    }
//////  
//////    DRAY_EXEC
//////    void get(int32 idx, RetValT &ret) const
//////    {
//////      QueryCat::get<RetValT>((const void **) m_ptrs, idx, ret, 0);
//////    }
//////  
//////  };
//////  
//////  namespace detail
//////  {
//////  // Cheat to use Vec::operator[]() or Matrix::get/set_col().
//////  template <typename DestT, typename SrcT>
//////  struct ColumnCopy
//////  {
//////    DRAY_EXEC
//////    static void cpy(DestT &dest, int32 d_idx, const SrcT &src, int32 s_idx);
//////  };
//////  
//////  // Vec <- Vec
//////  template <typename T, int32 DestS, int32 SrcS>
//////  struct ColumnCopy<Vec<T,DestS>, Vec<T,SrcS>>
//////  {
//////    DRAY_EXEC
//////    static void cpy(Vec<T,DestS> &dest, int32 d_idx, const Vec<T,SrcS> &src, int32 s_idx)
//////    {
//////      dest[d_idx] = src[s_idx];
//////    }
//////  };
//////  
//////  // Matrix <- Matrix
//////  template <typename T, int32 DestR, int32 DestC, int32 SrcR, int32 SrcC>
//////  struct ColumnCopy<Matrix<T,DestR,DestC>, Matrix<T,SrcR,SrcC>>
//////  {
//////    DRAY_EXEC
//////    static void cpy(Matrix<T,DestR,DestC> &dest, int32 d_idx, const Matrix<T,SrcR,SrcC> &src, int32 s_idx)
//////    {
//////      dest.set_col(d_idx, src.get_col(s_idx));
//////    }
//////  };
//////  
//////  }   // namespace detail.
//////  
//////  
//////  // Base case of the template-binary tree concatenation.
//////  template <int32 RefDim, typename ArrayCompT>
//////  struct QueryCatBase
//////  {
//////    static constexpr int32 ref_dim = RefDim;
//////    static constexpr int32 num_q = 1;
//////  
//////    ArrayCompT *m_ptr;
//////  
//////    DRAY_EXEC
//////    void get(int32 idx, ArrayCompT &ret) const
//////    {
//////      ret = m_ptr[idx];
//////    }
//////  
//////    DRAY_EXEC
//////    void set(int32 idx, ArrayCompT &val)
//////    {
//////      //TODO
//////    }
//////  
//////    template <typename RetValTT>
//////    DRAY_EXEC
//////    static void get(const void **ptrs, int32 idx, RetValTT &ret, int32 ret_offset)
//////    {
//////      // Single pointer to single array. num_q == 1. Get val in array at idx.
//////      const ArrayCompT val = ((const ArrayCompT *) *ptrs)[idx];
//////  
//////      // Over ref_dims of array component type.
//////      for (int32 in_dim = 0, out_dim = ret_offset; in_dim < ref_dim; in_dim++, out_dim++)
//////      {
//////        detail::ColumnCopy<RetValTT,ArrayCompT>::cpy(ret, out_dim, val, in_dim);
//////        //ret.set_col(out_dim, val.get_col(in_dim));
//////      }
//////    }
//////  };
//////  
//////  
//////  
//////  
//////  
//////  
//////  template <typename T, int32 PhysDim, int32 RefDim, int32 NumQ>
//////  struct ElTransCmpdQuery
//////  {
//////    static constexpr int32 phys_dim = PhysDim;
//////    static constexpr int32 ref_dim = RefDim;
//////    static constexpr int32 num_q = NumQ;
//////  
//////    virtual void perform(const Array<int32> &active_idx) = 0;
//////  
//////    const QuerySum<Vec<T,phys_dim>, num_q> sum_val_const()
//////    {
//////      QuerySum<Vec<T,phys_dim>, num_q> sum_f;
//////      sum_val_const(sum_f.m_ptrs, 0);
//////      return sum_f;
//////    }
//////  
//////    template <typename QCatT>
//////    const QCatT cat_deriv_const()
//////    {
//////      QCatT cat_f;
//////      cat_deriv_const(cat_f.m_ptrs, 0);
//////      return cat_f;
//////    }
//////  
//////    template <typename QCatT>
//////    const QCatT cat_ref_pt_const()
//////    {
//////      QCatT cat_f;
//////      cat_ref_pt_const(cat_f.m_ptrs, 0);
//////      return cat_f;
//////    }
//////  
//////    template <typename QCatT>
//////    QCatT cat_ref_pt()
//////    {
//////      QCatT cat_f;
//////      cat_ref_pt(cat_f.m_ptrs, 0);
//////      return cat_f;
//////    }
//////  
//////    virtual void sum_val_const(const Vec<T,phys_dim> * val_ptrs[], int32 offset) = 0;
//////    virtual void cat_deriv_const(const void * deriv_ptrs[], int32 offset) = 0;
//////    virtual void cat_ref_pt_const(const void * ref_pt_ptrs[], int32 offset) = 0;
//////    virtual void cat_ref_pt(void * ref_pt_ptrs[], int32 offset) = 0;
//////  
//////  ////  PtrBundleConst<num_q> get_val_ptrs_const() const;
//////  ////  PtrBundleConst<num_q> get_deriv_ptrs_const() const;
//////  ////  PtrBundleConst<num_q> get_ref_pt_ptrs_const() const;
//////  ////  PtrBundle<num_q> get_ref_pt_ptrs() = 0;
//////  ////
//////  ////  DRAY_EXEC
//////  ////  static void get_val(void * bundle_val[], int32 q_idx, Vec<T,phys_dim> &val)
//////  ////  {
//////  ////    val = ((Vec<T,phys_dim>*) bundle_val[0]) [q_idx];
//////  ////  }
//////  ////  DRAY_EXEC
//////  ////  static Vec<T,phys_dim> get_val(void ** bundle_val, int32 q_idx)
//////  ////  {
//////  ////    return ((Vec<T,phys_dim>*) bundle_val[0]) [q_idx];
//////  ////  }
//////  ////
//////  ////  DRAY_EXEC
//////  ////  static void get_deriv(void * bundle_deriv[], int32 q_idx, Matrix<T,phys_dim,ref_dim> &deriv)
//////  ////  {
//////  ////    //TODO
//////  ////  }
//////  ////
//////  ////  DRAY_EXEC
//////  ////  static void get_ref(void * bundle_ref[], int32 q_idx, Vec<T,ref_dim> &ref_pt)
//////  ////  {
//////  ////    //TODO
//////  ////  }
//////  ////
//////  ////  DRAY_EXEC
//////  ////  static void set_ref(void *bundle_ref[], int32 q_idx, const Vec<T, ref_dim> &ref_pt)
//////  ////  {
//////  ////    //TODO
//////  ////  }
//////  };
//////  
//////  // Pair (binary tree) of ElTransQueries
//////  template <typename T, typename QTypeA, typename QTypeB>
//////  struct ElTransQueryPair : public ElTransCmpdQuery<T, QTypeA::phys_dim, QTypeA::ref_dim + QTypeB::ref_dim, QTypeA::num_q + QTypeB::num_q>
//////  //struct ElTransQueryPair
//////  {
//////    //static constexpr int32 phys_dim = QTypeA::phys_dim;
//////  
//////    QTypeA query_a;
//////    QTypeB query_b;
//////  
//////    void perform(const Array<int32> &active_idx)
//////    {
//////      query_a.perform;
//////      query_b.perform;
//////    }
//////  
//////   //// const QueryCat<ref_dim, Matrix<T,phys_dim,ref_dim>> cat_deriv_const()
//////   //// {
//////   ////   QueryCatBase<ref_dim, Matrix<T,phys_dim,ref_dim>> cat_f;
//////   ////   cat_deriv_const(cat_f, 0);
//////   ////   return cat_f;
//////   //// }
//////  
//////   //// const QueryCat<ref_dim, Vec<T,ref_dim>> cat_ref_pt_const()
//////   //// {
//////   ////   QueryCatBase<ref_dim, Vec<T,ref_dim>> cat_f;
//////   ////   cat_ref_pt_const(cat_f, 0);
//////   ////   return cat_f;
//////   //// }
//////  
//////   //// QueryCat<ref_dim, Vec<T,ref_dim>> cat_ref_pt()
//////   //// {
//////   ////   QueryCatBase<ref_dim, Vec<T,ref_dim>> cat_f;
//////   ////   cat_ref_pt(cat_f, 0);
//////   ////   return cat_f;
//////   //// }
//////  
//////    virtual void sum_val_const(const Vec<T,ElTransQueryPair::phys_dim> * val_ptrs[], int32 offset)
//////    {
//////      query_a.sum_val_const(val_ptrs, offset);
//////      query_b.sum_val_const(val_ptrs, offset + QTypeA::num_q);
//////    }
//////  
//////    virtual void cat_deriv_const(const void * deriv_ptrs[], int32 offset)
//////    {
//////      query_a.cat_deriv_const(val_ptrs, offset);
//////      query_b.cat_deriv_const(val_ptrs, offset + QTypeA::num_q);
//////    } 
//////  
//////    virtual void cat_ref_pt_const(const void * ref_pt_ptrs[], int32 offset)
//////    {
//////      query_a.cat_ref_pt_const(val_ptrs, offset);
//////      query_b.cat_ref_pt_const(val_ptrs, offset + QTypeA::num_q);
//////    }
//////  
//////    virtual void cat_ref_pt(void * ref_pt_ptrs[], int32 offset)
//////    {
//////      query_a.cat_ref_pt(val_ptrs, offset);
//////      query_b.cat_ref_pt(val_ptrs, offset + QTypeA::num_q);
//////    }
//////  };
//////  
//////  template <typename T, typename A, typename B>
//////  template auto ElTransQueryPair<T,A,B>::cat_deriv_const<QueryCat<ref_dim, Matrix<T,phys_dim,ref_dim>>>();
//////  
//////  
//////  /*
//////   * An ElTransQuery returns a value/derivative that may be part of a larger query.
//////   *
//////   * The methods which actually return values must do so per array column, across a struct of arrays.
//////   * Therefore they must be DRAY_EXEC and 
//////   */
//////  
//////  
//////  
//////  
//////  template <typename T, typename ElTransType>
//////  struct ElTransQuery : public ElTransCmpdQuery<T, ElTransType::PhysDim, ElTransType::RefDim, 1>
//////  //class ElTransQuery
//////  {
//////    ///static constexpr int32 phys_dim = ElTransType::PhysDim;
//////    ///static constexpr int32 ref_dim = ElTransType::RefDim;
//////    ///static constexpr int32 num_q = 1;  //self, this is leaf.
//////  
//////    using q_sum_t = QuerySum<Vec<T,phys_dim>, 1>;
//////  
//////    template <typename ArrayCompType>
//////    using q_cat_t = QueryCatBase<ref_dim, ArrayCompType>;
//////  
//////    Array<int32> m_elt_id;
//////    Array<Vec<T,ref_dim>> m_ref_pts;
//////    Array<Vec<T,phys_dim>> m_result_val;
//////    Array<Matrix<T,phys_dim,ref_dim>> m_result_deriv;
//////  
//////    void perform(const Array<int32> &active_idx)
//////    {
//////      //TODO
//////    }
//////  
//////  
//////    ////virtual const QueryCatBase<ref_dim, Matrix<T,phys_dim,ref_dim>> cat_deriv_const()
//////    ////{
//////    ////  QueryCatBase<ref_dim, Matrix<T,phys_dim,ref_dim>> cat_f;
//////    ////  cat_deriv_const(cat_f, 0);
//////    ////  return cat_f;
//////    ////}
//////  
//////    ////virtual const QueryCatBase<ref_dim, Vec<T,ref_dim>> cat_ref_pt_const()
//////    ////{
//////    ////  QueryCatBase<ref_dim, Vec<T,ref_dim>> cat_f;
//////    ////  cat_ref_pt_const(cat_f, 0);
//////    ////  return cat_f;
//////    ////}
//////  
//////    ////virtual QueryCatBase<ref_dim, Vec<T,ref_dim>> cat_ref_pt()
//////    ////{
//////    ////  QueryCatBase<ref_dim, Vec<T,ref_dim>> cat_f;
//////    ////  cat_ref_pt(cat_f, 0);
//////    ////  return cat_f;
//////    ////}
//////  
//////    virtual void sum_val_const(const Vec<T,phys_dim> * val_ptrs[], int32 arr)
//////    {
//////      val_ptrs[arr] = m_result_val.get_device_ptr_const();
//////    }
//////  
//////    virtual void cat_deriv_const(const void * deriv_ptrs[], int32 arr)
//////    {
//////      deriv_ptrs[arr] = m_result_deriv.get_device_ptr_const();
//////    } 
//////  
//////    virtual void cat_ref_pt_const(const void * ref_pt_ptrs[], int32 arr)
//////    {
//////      ref_pt_ptrs[arr] = m_ref_pts.get_device_ptr_const();
//////    }
//////  
//////    virtual void cat_ref_pt(void * ref_pt_ptrs[], int32 arr)
//////    {
//////      ref_pt_ptrs[arr] = m_ref_pts.get_device_ptr();
//////    }
//////  };
//////    //template virtual auto cat_deriv_const<QueryCatBase<ref_dim, Matrix<T,phys_dim,ref_cim>>>();
//////  
//////  

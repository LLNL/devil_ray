#ifndef DRAY_BERSTEIN_BASIS_HPP
#define DRAY_BERSTEIN_BASIS_HPP

#include <dray/binomial.hpp>          // For BinomRow, to help BernsteinBasis.
#include <dray/vec.hpp>

#include <string.h>  // memcpy()

namespace dray
{

template <typename T, int32 RefDim>
struct BernsteinBasis
{
  // Internals
  int32 p;
  T *m_aux_mem_ptr;

  // Public
  DRAY_EXEC void init_shape(int32 _p, T *aux_mem_ptr) { p = _p; m_aux_mem_ptr = aux_mem_ptr; }

  static constexpr int32 ref_dim = RefDim;
  DRAY_EXEC int32 get_el_dofs() const { return pow(p+1, RefDim); }

  DRAY_EXEC void set_aux_mem_ptr(T *aux_mem_ptr) { m_aux_mem_ptr = aux_mem_ptr; }

    // The number of auxiliary elements needed for member aux_mem.
    // For each reference dim, need a row for values and a row for derivatives.
    // Can compute tensor-product on the fly from these rows.
  static int32 get_aux_req(int32 p) { return 2 * RefDim * (p+1); }
  DRAY_EXEC int32 get_aux_req() const { return 2 * RefDim * (p+1); }
  DRAY_EXEC static bool is_aux_req() { return true; }

    // Linear combination of value functions, and linear combinations of derivative functions.
    // This is to evaluate a transformmation using a given set of control points at a given reference points.
  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo(const Vec<T,RefDim> &xyz,
                              const CoeffIterType &coeff_iter,
                              Vec<T,PhysDim> &result_val,
                              Vec<Vec<T,PhysDim>,RefDim> &result_deriv);

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo_old(const Vec<T,RefDim> &xyz,
                              const CoeffIterType &coeff_iter,
                              Vec<T,PhysDim> &result_val,
                              Vec<Vec<T,PhysDim>,RefDim> &result_deriv);
 
  DRAY_EXEC static bool is_inside(const Vec<T,RefDim> ref_pt)
  {
    for (int32 rdim = 0; rdim < RefDim; rdim++)
      if (!(0 <= ref_pt[rdim] && ref_pt[rdim] <= 1))     //TODO
        return false;
    return true;
  }

    // If just want raw shape values/derivatives,
    // stored in memory, to do something with them later:
  ////DRAY_EXEC void calc_shape_dshape(const Vec<T,RefDim> &ref_pt, T *shape_val, Vec<T,RefDim> *shape_deriv) const;   //TODO

};  // BernsteinBasis


namespace detail_BernsteinBasis
{
  // Helper functions to access the auxiliary memory space.
  DRAY_EXEC static int32 aux_mem_val_offset(int32 p, int32 rdim) { return (2*rdim) * (p+1); }
  DRAY_EXEC static int32 aux_mem_deriv_offset(int32 p, int32 rdim) { return (2*rdim + 1) * (p+1); }

  // Bernstein evaluator using pow(). Assumes binomial coefficient is the right one (p choose k).
  template <typename T>
  DRAY_EXEC
  static void calc_shape_dshape_1d_single(const int32 p, const int32 k, const T x, const int32 bcoeff, T &u, T &d)
  {
    if (p == 0)
    {
      u = 1.;
      d = 0.;
    }
    else
    {
      u = bcoeff * pow(x, k) * pow(1-x, p-k);
      d = (k == 0 ? -p * pow(1-x, p-1) :
           k == p ?  p * pow(x, p-1)    :
                    (k - p*x) * pow(x, k-1) * pow(1-x, p-k-1)) * bcoeff;
    }
  }

  // Bernstein evaluator adapted from MFEM.
  template <typename T>
  DRAY_EXEC
  static void calc_shape_dshape_1d(const int32 p, const T x, const T y, T *u, T *d)
  {
    if (p == 0)
    {
       u[0] = 1.;
       d[0] = 0.;
    }
    else
    {
      // Assume that binomial coefficients are already sitting in the arrays u[], d[].
      const double xpy = x + y, ptx = p*x;
      double z = 1.;

      int i;
      for (i = 1; i < p; i++)
      {
         //d[i] = b[i]*z*(i*xpy - ptx);
         d[i] = d[i]*z*(i*xpy - ptx);
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

  template <typename T>
  DRAY_EXEC static void calc_shape_1d(const int32 p, const T x, const T y, T *u)
  {
    if (p == 0)
    {
       u[0] = 1.;
    }
    else
    {
      // Assume that binomial coefficients are already sitting in the array u[].
      double z = 1.;
      int i;
      for (i = 1; i < p; i++)
      {
         z *= x;
         u[i] = u[i]*z;
      }
      u[p] = z*x;
      z = 1.;
      for (i--; i > 0; i--)
      {
         z *= y;
         u[i] *= z;
      }
      u[0] = z*y;
    }
  }

  template <typename T>
  DRAY_EXEC static void calc_dshape_1d(const int32 p, const T x, const T y, T *d)
  {
    if (p == 0)
    {
       d[0] = 0.;
    }
    else
    {
      // Assume that binomial coefficients are already sitting in the array d[].
      const double xpy = x + y, ptx = p*x;
      double z = 1.;

      int i;
      for (i = 1; i < p; i++)
      {
         d[i] = d[i]*z*(i*xpy - ptx);
         z *= x;
      }
      d[p] = p*z;
      z = 1.;
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
      }
      d[0] = -p*z;
    }
  }

}  // namespace detail_BernsteinBasis


// TODO after re-reverse lex, use recursive templates and clean this file up.

template <typename T, int32 RefDim>
  template <typename CoeffIterType, int32 PhysDim>
DRAY_EXEC void
BernsteinBasis<T,RefDim>::linear_combo(
    const Vec<T,RefDim> &xyz,
    const CoeffIterType &coeff_iter,
    Vec<T,PhysDim> &result_val,
    Vec<Vec<T,PhysDim>,RefDim> &result_deriv)
{
  // The simple way, using pow() and not aux_mem_ptr.

  // Initialize output parameters.
  result_val = 0;
  for (int32 rdim = 0; rdim < RefDim; rdim++)
    result_deriv[rdim] = 0;

  const int32 pp1 = p+1;

  //
  // Accumulate the tensor product components.
  // Read each control point once.
  //

  // Set up index formulas.
  // First coordinate is outermost, encompasses all. Last is innermost, encompases (p+1).
  int32 stride[RefDim];
  stride[RefDim - 1] = 1;
  for (int32 rdim = RefDim - 2; rdim >= 0; rdim--)
  {
    stride[rdim] = (pp1) * stride[rdim+1];
  }
  int32 el_dofs = (pp1) * stride[0];

  int32 ii[RefDim];
  T shape_val[RefDim];
  T shape_deriv[RefDim];

  // Construct the binomial coefficients (part of the shape functions).
  BinomRowIterator binom_coeff[RefDim];
  for (int32 rdim = 0; rdim < RefDim; rdim++)
  {
    ii[rdim] = 0;
    binom_coeff[rdim].construct(p);
    detail_BernsteinBasis::calc_shape_dshape_1d_single(p, 0, xyz[rdim], *binom_coeff[rdim],
        shape_val[rdim], shape_deriv[rdim]);
  }

  // Iterate over degrees of freedom, i.e., iterate over control point values.
  for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
  {
    // Compute index and new shape values.
    for (int32 rdim_in = 0; rdim_in < RefDim; rdim_in++)
    {
      int32 tmp_ii = (dof_idx / stride[rdim_in]) % (pp1);
      if (tmp_ii != ii[rdim_in])  // On the edge of the next step in ii[rdim_in].
      {
        ii[rdim_in] = tmp_ii;
        binom_coeff[rdim_in].next();
        detail_BernsteinBasis::calc_shape_dshape_1d_single(p, ii[rdim_in], xyz[rdim_in], *binom_coeff[rdim_in],
            shape_val[rdim_in], shape_deriv[rdim_in]);
      }
    }

    // Compute tensor product shape.
    T t_shape_val = shape_val[0];
    for (int32 rdim_in = 1; rdim_in < RefDim; rdim_in++)
      t_shape_val *= shape_val[rdim_in];

    // Multiply control point value, accumulate value.
    const Vec<T,PhysDim> ctrl_val = coeff_iter[dof_idx];
    result_val +=  ctrl_val * t_shape_val;

    for (int32 rdim_out = 0; rdim_out < RefDim; rdim_out++)   // Over the derivatives.
    {
      T t_shape_deriv = shape_deriv[rdim_out];
      int32 rdim_in;
      for (rdim_in = 0; rdim_in < rdim_out; rdim_in++)    // Over the tensor dimensions.
        t_shape_deriv *= shape_val[rdim_in];
      for ( ++rdim_in; rdim_in < RefDim; rdim_in++)       // Over the tensor dimensions.
        t_shape_deriv *= shape_val[rdim_in];

      // Multiply control point value, accumulate value.
      result_deriv[rdim_out] +=  ctrl_val * t_shape_deriv;
    }
  }
}


template <typename T, int32 RefDim>
  template <typename CoeffIterType, int32 PhysDim>
DRAY_EXEC void
BernsteinBasis<T,RefDim>::linear_combo_old(
    const Vec<T,RefDim> &xyz,
    const CoeffIterType &coeff_iter,
    Vec<T,PhysDim> &result_val,
    Vec<Vec<T,PhysDim>,RefDim> &result_deriv)
{
  // Initialize output parameters.
  result_val = 0;
  for (int32 rdim = 0; rdim < RefDim; rdim++)
    result_deriv[rdim] = 0;

  const int32 pp1 = p+1;

  // Make names for the rows of auxiliary memory.
  T* val_i[RefDim];
  T* deriv_i[RefDim];
  for (int32 rdim = 0; rdim < RefDim; rdim++)
  {
    val_i[rdim] = m_aux_mem_ptr + detail_BernsteinBasis::aux_mem_val_offset(p,rdim);
    deriv_i[rdim] = m_aux_mem_ptr + detail_BernsteinBasis::aux_mem_deriv_offset(p,rdim);
  }

  // The first two rows will be used specially.
  T* &val_0 = val_i[0];
  T* &deriv_0 = deriv_i[0];

  //
  // Populate shape values and derivatives.
  //

  // Fill the first two rows with binomial coefficients.
  BinomRow<T>::fill_single_row(p, val_0);
  memcpy(deriv_0, val_0, pp1 * sizeof(T));

  // Compute shape values and derivatives for latter dimensions.
  for (int32 rdim = 1; rdim < RefDim; rdim++)
  {
    // Copy binomial coefficients.
    memcpy(val_i[rdim], val_0, pp1 * sizeof(T));
    memcpy(deriv_i[rdim], val_0, pp1 * sizeof(T));

    // Compute shape values and derivatives.
    const T x_i = xyz[rdim];
    detail_BernsteinBasis::calc_shape_1d<T>(p, x_i, 1. - x_i, val_i[rdim]);
    detail_BernsteinBasis::calc_dshape_1d<T>(p, x_i, 1. - x_i, deriv_i[rdim]);
  }

  // Compute shape values and derivatives for first dimension.
  const T x_0 = xyz[0];
  detail_BernsteinBasis::calc_shape_1d<T>(p, x_0, 1. - x_0, val_0);
  detail_BernsteinBasis::calc_dshape_1d<T>(p, x_0, 1. - x_0, deriv_0);

  //
  // Accumulate the tensor product components.
  // Read each control point once.
  //

  // Set up index formulas.
  // First coordinate is outermost, encompasses all. Last is innermost, encompases (p+1).
  int32 stride[RefDim];
  stride[RefDim - 1] = 1;
  for (int32 rdim = RefDim - 2; rdim >= 0; rdim--)
  {
    stride[rdim] = (pp1) * stride[rdim+1];
  }
  int32 el_dofs = (pp1) * stride[0];

  // Iterate over degrees of freedom, i.e., iterate over control point values.
  for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
  {
    int32 ii[RefDim];

    T t_shape_val = 1.;
    T shape_val_1d[RefDim];  // Cache the values, we'll reuse multiple times in the derivative computation.
    for (int32 rdim_in = 0; rdim_in < RefDim; rdim_in++)
    {
      ii[rdim_in] = (dof_idx / stride[rdim_in]) % (pp1);
      shape_val_1d[rdim_in] = val_i[rdim_in][ ii[rdim_in] ];
      t_shape_val *= shape_val_1d[rdim_in];
    }

    // Multiply control point value, accumulate value.
    const Vec<T,PhysDim> ctrl_val = coeff_iter[dof_idx];
    result_val +=  ctrl_val * t_shape_val;

    for (int32 rdim_out = 0; rdim_out < RefDim; rdim_out++)   // Over the derivatives.
    {
      T t_shape_deriv = 1.;
      int32 rdim_in;
      for (rdim_in = 0; rdim_in < rdim_out; rdim_in++)    // Over the tensor dimensions.
        t_shape_deriv *= val_i[rdim_in][ ii[rdim_in] ];
      t_shape_deriv *= deriv_i[rdim_out][ ii[rdim_out] ];
      for ( ++rdim_in; rdim_in < RefDim; rdim_in++)       // Over the tensor dimensions.
        t_shape_deriv *= val_i[rdim_in][ ii[rdim_in] ];

      // Multiply control point value, accumulate value.
      result_deriv[rdim_out] +=  ctrl_val * t_shape_deriv;
    }
  }
}

}// namespace dray

#endif


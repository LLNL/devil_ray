#include "gtest/gtest.h"
#include <dray/high_order_shape.hpp>
#include <dray/math.hpp>

#include <iostream>



// Discrete uniform distribution.
template <typename T, int32 D, int32 DOF>
struct DummyUniformShape
{
  ///DRAY_EXEC void operator()(const Vec<T,D> &ref_pt, Vec<T,DOF> &shape_out) const
  void operator()(const Vec<T,D> &ref_pt, Vec<T,DOF> &shape_out) const
  {
    shape_out = static_cast<T>(1.f) / DOF;
  }
};



TEST(dray_test, dray_newton_solve)
{
  constexpr int DOF = dray::IntPow<2+1,3>::val;
  
  const int num_elts = 2;
  constexpr int num_field_comp = 1;  // SCALAR field.

  dray::FunctionCtrlPoints<float, num_field_comp, DOF>  scalar_field;

  scalar_field.m_values.resize(num_elts*DOF); // In this case, no control points are shared among different elements.
  const float *values_ptr = (float *) scalar_field.m_values.get_host_ptr_const();
  for (int ctrl_idx = 0; ctrl_idx < num_elts*DOF; ctrl_idx++)
  {
    values_ptr[ctrl_idx] = static_cast<float>(ctrl_idx);
  }

  scalar_field.m_ctrl_idx = array_counting(num_elts*DOF, 0,1);

  //typedef dray::detail::BernsteinShape<float,3,2> BShape_3_2;
  //dray::Vec<float, 4> shape_here = fcp.eval(BShape_3_2(), ref_pts);

  dray::Array<dray::Vec<float,3>> ref_pts;
  ref_pts.resize(num_elts);
  const Vec<float,3> ref_center = {0.5, 0.5, 0.5};
  array_memset_vec(ref_pts, ref_center);

  // Evaluate
  dray::Array<dray::Vec<float,1>> elt_vals;
  elt_vals.resize(num_elts);
  dray::Array<dray::Vec<float,num_field_comp>> elt_vals;
  elt_vals = scalar_field.eval(DummyUniformShape<float, 3, DOF);

  // Report
  elt_vals.summary();
}

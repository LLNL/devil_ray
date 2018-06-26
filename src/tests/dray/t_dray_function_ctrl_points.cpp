#include "gtest/gtest.h"
#include <dray/high_order_shape.hpp>
#include <dray/arrayvec.hpp>
#include <dray/vec.hpp>
#include <dray/math.hpp>

#include <iostream>


TEST(dray_test, dray_function_ctrl_points)
{

  /// dray::detail::BernsteinShape<float,3,2> bshape_3_2;
  /// dray::detail::BernsteinShape<float,1,0> bshape_1_0;

  /// dray::Vec<float,1> ref_pt;
  /// dray::Vec<float,1> shape_pt;
  /// bshape_1_0(ref_pt, shape_pt);

  /// typename dray::ArrayVec<float,1>::type scalar_array;
  /// scalar_array.resize(10);

 
  constexpr int DOF = dray::IntPow<2+1,3>::val;
  
  const int num_elts = 5;
  constexpr int num_field_comp = 1;  // SCALAR field.

  dray::FunctionCtrlPoints<float, num_field_comp, DOF>  scalar_field;

  // There are enough control points for every element to have DOF distinct control points.
  // Set the values to increase linearly with index.
  scalar_field.m_values.resize(num_elts*DOF); 
  float *values_ptr = (float *) scalar_field.m_values.get_host_ptr();
  for (int ctrl_idx = 0; ctrl_idx < num_elts*DOF; ctrl_idx++)
  {
    values_ptr[ctrl_idx] = static_cast<float>(ctrl_idx);
  }

  // In this case, no control points are shared among different elements.
  scalar_field.m_ctrl_idx.resize(num_elts*DOF);
  int *ctrl_idx_ptr = scalar_field.m_ctrl_idx.get_host_ptr();
  for (int ii = 0; ii < num_elts*DOF; ii++)
  {
    ctrl_idx_ptr[ii] = ii;
  }

  //typedef dray::detail::BernsteinShape<float,3,2> BShape_3_2;
  //dray::Vec<float, 4> shape_here = fcp.eval(BShape_3_2(), ref_pts);

  // Initialize a list of reference points, 1 per element. All are at center.
  const dray::Vec<float,3> ref_center = {0.5, 0.5, 0.5};
  dray::Array<dray::Vec<float,3>> ref_pts;
  ref_pts.resize(num_elts);
  dray::Vec<float,3> *ref_pts_ptr = ref_pts.get_host_ptr();
  for (int ii = 0; ii < num_elts; ii++)
  {
    ref_pts_ptr[ii] = ref_center;
  }

  // Evaluate.
  // Ctrl point values increase from 0 in steps of 1, and each element
  // has 27 DOFs. Therefore, the 0th element has mean value 13, and the
  // element mean values increase in steps of 27.
  typedef dray::detail::DummyUniformShape<float, 3, DOF> DummyFunctor;
  dray::ArrayVec<float, num_field_comp> elt_vals;
  elt_vals = scalar_field.eval<DummyFunctor,3>(DummyFunctor(), ref_pts);

  // Report
  elt_vals.summary();
}

#include "gtest/gtest.h"
#include <dray/high_order_shape.hpp>
#include <dray/array.hpp>
#include <dray/arrayvec.hpp>
#include <dray/vec.hpp>
#include <dray/math.hpp>
#include <dray/binomial.hpp>

#include <iostream>


TEST(dray_test, dray_high_order_shape)
{
  constexpr int DOF = dray::IntPow<2+1,3>::val;

  //--- Test classes ---//
  ///dray::BernsteinShape<float, 2> bshape = {4};
  ///std::cout << "bshape el_dofs = " << bshape.get_el_dofs() << std::endl;


  //--- Test linear shape evaluator---//

  // Make 3 queries on a trilinear element.
  int _active_idx[3] = {0, 1, 2};
  dray::Array<int> active_idx(_active_idx, 3);
  dray::Vec<float,3> _ref_pts[3] = {{0.,0.,0.}, {.5,.5,.5}, {.25,.75,1.}};
  dray::Array<dray::Vec<float,3>> ref_pts(_ref_pts, 3);

  dray::LinearShape<float, 3> lshape;
  //void calc_shape_dshape( const Array<int32> &active_idx,
  //                        const ArrayVec<T,RefDim> &ref_pts,
  //                        Array<T> &shape_val,
  //                        ArrayVec<T,RefDim> &shape_deriv) const;


  //--- Test binomial coefficients ---//
  std::cout << dray::Binom<5,0>::val << " "
            << dray::Binom<5,1>::val << " "
            << dray::Binom<5,2>::val << " "
            << dray::Binom<5,3>::val << " "
            << dray::Binom<5,4>::val << " "
            << dray::Binom<5,5>::val << std::endl;


  const float *binom_row = dray::BinomRow<float, 8>::get_static();
  std::cout << binom_row[0] << " "
            << binom_row[1] << " "
            << binom_row[2] << " "
            << binom_row[3] << " "
            << binom_row[4] << " "
            << binom_row[5] << " "
            << binom_row[6] << " "
            << binom_row[7] << " "
            << binom_row[8] << std::endl;

  dray::BinomRow<float, 6> binom_row_obj;
  binom_row = binom_row_obj.get();
  std::cout << binom_row[0] << " "
            << binom_row[1] << " "
            << binom_row[2] << " "
            << binom_row[3] << " "
            << binom_row[4] << " "
            << binom_row[5] << " "
            << binom_row[6] << std::endl;

  //--- Test Bernstein shape evaluator ---//

  /// dray::detail::BernsteinShape<float,1,0> bshape_1_0;  // not using, just see if compiles.

  /// dray::detail::BernsteinShape<float,3,2> bshape_3_2;  // Quadratic volume, 27 dof
  /// dray::Vec<float,DOF> shape_pt;

  /// bshape_3_2.calc_shape({0,0,0}, shape_pt);   std::cout << shape_pt << std::endl;
  /// bshape_3_2.calc_shape({1,0,1}, shape_pt);   std::cout << shape_pt << std::endl;
  /// bshape_3_2.calc_shape({.75,.25,1.}, shape_pt);   std::cout << shape_pt << std::endl;
  /// bshape_3_2.calc_shape({.5,.5,1.}, shape_pt);   std::cout << shape_pt << std::endl;
  /// bshape_3_2.calc_shape({.75,.75,.75}, shape_pt);   std::cout << shape_pt << std::endl;
  /// bshape_3_2.calc_shape({.5,.5,.5}, shape_pt);   std::cout << shape_pt << std::endl;



  //--- Test FunctionCtrlPoint ---//

  /// ///constexpr int DOF = dray::IntPow<2+1,3>::val;
  /// 
  /// const int num_elts = 5;
  /// constexpr int num_field_comp = 1;  // SCALAR field.

  /// typedef dray::detail::DummyUniformShape<float, 3, DOF> DummyFunctor;
  /// dray::FunctionCtrlPoints<float, num_field_comp, DOF, DummyFunctor, 3>  scalar_field;

  /// // There are enough control points for every element to have DOF distinct control points.
  /// // Set the values to increase linearly with index.
  /// scalar_field.m_values.resize(num_elts*DOF); 
  /// float *values_ptr = (float *) scalar_field.m_values.get_host_ptr();
  /// for (int ctrl_idx = 0; ctrl_idx < num_elts*DOF; ctrl_idx++)
  /// {
  ///   values_ptr[ctrl_idx] = static_cast<float>(ctrl_idx);
  /// }

  /// // In this case, no control points are shared among different elements.
  /// scalar_field.m_ctrl_idx.resize(num_elts*DOF);
  /// int *ctrl_idx_ptr = scalar_field.m_ctrl_idx.get_host_ptr();
  /// for (int ii = 0; ii < num_elts*DOF; ii++)
  /// {
  ///   ctrl_idx_ptr[ii] = ii;
  /// }

  /// //typedef dray::detail::BernsteinShape<float,3,2> BShape_3_2;
  /// //dray::Vec<float, 4> shape_here = fcp.eval(BShape_3_2(), ref_pts);

  /// // Initialize a list of reference points, 1 per element. All are at center.
  /// const dray::Vec<float,3> ref_center = {0.5, 0.5, 0.5};
  /// dray::Array<dray::Vec<float,3>> ref_pts;
  /// ref_pts.resize(num_elts);
  /// dray::Vec<float,3> *ref_pts_ptr = ref_pts.get_host_ptr();
  /// for (int ii = 0; ii < num_elts; ii++)
  /// {
  ///   ref_pts_ptr[ii] = ref_center;
  /// }

  /// // Evaluate.
  /// // Ctrl point values increase from 0 in steps of 1, and each element
  /// // has 27 DOFs. Therefore, the 0th element has mean value 13, and the
  /// // element mean values increase in steps of 27.
  /// dray::ArrayVec<float, num_field_comp> elt_vals;
  /// elt_vals = scalar_field.eval(DummyFunctor(), ref_pts);

  /// // Report
  /// elt_vals.summary();
}

// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include <dray/types.hpp>
#include <dray/synthetic/affine_radial.hpp>

#include <dray/Element/element.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/elem_ops.hpp>

#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/device_field.hpp>

#include <iostream>

TEST(dray_hessian, dray_zero_hessian)
{
  using dray::Float;

  const dray::Vec<int, 3> extents = {{4, 4, 4}};
  const dray::Vec<Float, 3> origin = {{0.0f, 0.0f, 0.0f}};
  const dray::Vec<Float, 3> radius = {{1.0f, 1.0f, 1.0f}};

  // Collection of hexs with no fields.
  dray::Collection collxn =
      dray::SynthesizeAffineRadial(extents, origin, radius)
      .synthesize();

  using ScalarFElemT = dray::Element<3, 1, dray::Tensor, dray::General>;

  // Add a constant field.
  const dray::Vec<Float, 1> constant = {{5.0f}};
  const std::string field_name = "VeryConstant";
  for (dray::DataSet &ds : collxn.domains())
    ds.add_field(std::make_shared<dray::Field<ScalarFElemT>>(
          dray::Field<ScalarFElemT>::uniform_field(ds.topology()->cells(), constant, field_name)));

  dray::DataSet &data_set = *collxn.domains().begin();

  // We know what type the field is because we just made it.
  dray::Field<ScalarFElemT> &field = * (dray::Field<ScalarFElemT> *)(data_set.field(field_name));
  dray::DeviceField<ScalarFElemT> dfield(field);

  // Note: DeviceField should only be used on the device. This code will not work on cuda.
  {
    dray::Matrix<dray::Vec<Float, 1>, 3, 3> H_elem0;

    std::cerr << "Warning: If attempting to run with cuda, expect a segfault.\n";

    const dray::Vec<Float, 3> ref_coord = {{0.1, 0.3, 0.6}};
    dray::eops::eval_hessian(dray::ShapeHex(),
                             dfield.get_order_policy(),
                             (dray::ReadDofPtr<dray::Vec<Float, 1>>) dfield.get_elem(0).read_dof_ptr(),
                             ref_coord,
                             H_elem0);

    std::cout << "H_elem0:  \n" << H_elem0 << "\n";
  }
}


TEST(dray_hessian, dray_constant_hessian)
{
  using dray::Float;

  const dray::Vec<int, 3> extents = {{4, 4, 4}};
  const dray::Vec<Float, 3> origin = {{0.0f, 0.0f, 0.0f}};
  const dray::Vec<Float, 3> radius = {{1.0f, 1.0f, 1.0f}};
  const dray::Vec<Float, 3> range_radius = {{1.0f, 1.0f, -1.0f}};

  const std::string field_name = "perfection";

  dray::Collection collxn =
      dray::SynthesizeAffineRadial(extents, origin, radius)
      .equip(field_name, range_radius)
      .synthesize();

  dray::DataSet &data_set = *collxn.domains().begin();

  using ScalarFElemT = dray::Element<3, 1, dray::Tensor, 2>;

  // I know this is the type of the field, just accept it.
  dray::Field<ScalarFElemT> &field = * (dray::Field<ScalarFElemT> *)(data_set.field(field_name));
  dray::DeviceField<ScalarFElemT> dfield(field);

  // Note: DeviceField should only be used on the device. This code will not work on cuda.
  {
    dray::Matrix<dray::Vec<Float, 1>, 3, 3> H_elem0;
    dray::Matrix<dray::Vec<Float, 1>, 3, 3> H_elem1;

    std::cerr << "Warning: If attempting to run with cuda, expect a segfault.\n";

    const dray::Vec<Float, 3> ref_coord = {{0.1, 0.3, 0.6}};
    dray::eops::eval_hessian(dray::ShapeHex(),
                             /*dfield.get_order_policy(),*/
                                 dray::OrderPolicy<dray::General>{2},  // Interim before specializations implemented.
                             (dray::ReadDofPtr<dray::Vec<Float, 1>>) dfield.get_elem(0).read_dof_ptr(),
                             ref_coord,
                             H_elem0);
    dray::eops::eval_hessian(dray::ShapeHex(),
                             /*dfield.get_order_policy(),*/
                                 dray::OrderPolicy<dray::General>{2},  // Interim before specializations implemented.
                             (dray::ReadDofPtr<dray::Vec<Float, 1>>) dfield.get_elem(1).read_dof_ptr(),
                             ref_coord,
                             H_elem1);

    std::cout << "H_elem0:  \n" << H_elem0 << "\n";
    std::cout << "H_elem1:  \n" << H_elem1 << "\n";
  }
}



//
// Pseudocode for formulas

/*
 *
 *
TEST(dray_hessian, dray_grad_mag_grad)
{
  using dray::Vec<Float, 1>;
  using dray::Vec<Float, 3>;
  using dray::Vec<Vec<Float, 3>, 3>;

  Vec<Float, 3> ref;

  // Evaluate Phi and derivatives.
  Vec<Vec<Float, 3>, 3> J; // each vec is col
  Vec<Float, 3> world = mesh_elem.eval_d(ref, J);
  Vec<Matrix<Float, 3, 3>, 3> D2_Phi = as_vec_of_matrix(mesh_elem.eval_hessian(ref));

  // Get the inverse-transpose of the Jacobian.
  bool inv_valid;
  1 2 3 | 4 5 6 | 7 8 9 -> 1 4 7 -> 1 2 3
                           2 5 8 -> 4 5 6
                           3 6 9 -> 7 8 9
  Vec<Vec<Float, 3>, 3> J;

  Matrix<Float, 3, 3> Jt = Matrix<Float, 3, 3>::transpose_matrix_from_col_major(J);
  MatrixInverse<Float, 3> Jt_inv(Jt, inv_valid);  // LU decomposition

  // Evaluate f (scalar field) and derivatives w.r.t. reference space.
  Vec<Vec<Float, 1>, 3> grad_f_ref;
  Vec<Float, 1> f = field_elem.eval_d(ref, grad_f_ref);
  Vec<Float, 3> D1_f_ref = squeeze(grad_f_ref); // ->  Vec<vec<1>> -> Vec
  Matrix<Vec<Float, 1>, 3, 3> D2_f_ref = field_elem.eval_hessian(ref);

  // 1st derivative of f (scalar field) w.r.t. world space.
  Vec<Float, 3> D1_f_world = Jt_inv * D1_f_ref;


  // ===========================================================================
  // The formula
  // ===========================================================================
  const Float grad_mag = D1_f_world.magnitude();

  const Vec<Float, 3> D1_grad_mag_world =
      Jt_inv * ( (D2_f_ref - dot(D2_Phi, D1_f_world)) * D1_f_world.normalized() );
  //
  //                         dot(vec<mat> vec<scalar>)
  //                         -----------------------
  //              (mat     -         mat            ) *  vec
  //   -------    -------------------------------------------------------------
  //   mat inv  *            vec
  //
  // ===========================================================================
}
*
*
*/




/*
 *
 *
TEST(dray_hessian, dray_vec_mag_grad)
{
  Vec<Float, 3> ref;

  // Evaluate Phi and Jacobian.
  Vec<Vec<Float, 3>, 3> J;
  Vec<Float, 3> world = mesh_elem.eval_d(ref, J);

  // Get the inverse-transpose of the Jacobian.
  bool inv_valid;
  Matrix<Float, 3, 3> Jt = Matrix<Float, 3, 3>::transpose_matrix_from_col_major(J);
  MatrixInverse<Float, 3> Jt_inv(Jt, inv_valid);  // LU decomposition

  // Evaluate v (vector field) and derivative w.r.t. reference space.
  Vec<Vec<Float, 3>, 3> D1_v_ref;
  Vec<Float, 3> v = field_elem.eval_d(ref, D1_v_ref);

  Matrix<Float, 3, 3> D1_v_ref_t =
      Matrix<Float, 3, 3>::transpose_matrix_from_col_major(D1_v_ref);

  // Derivative of v (vector field) w.r.t. world space.
  Matrix<Float, 3, 3> D1_v_world_t = Jt_inv * D1_v_ref_t;

  // ===========================================================================
  // Gradient of vector magnitude.
  // ===========================================================================
  Vec<Float, 3> vec_mag_grad = D1_v_world_t * v.normalized();
  // ===========================================================================
}
*
*
*/

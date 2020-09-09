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
#include <numeric>  //iota

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



TEST(dray_hessian, dray_grad_mag_grad)
{
  constexpr int SCALAR_COMP = 1;
  constexpr int REFERENCE_COMP = 3;
  constexpr int WORLD_COMP = 3;

  using dray::Float;
  using dray::Vec;
  using dray::Matrix;

  constexpr int NPE_CUBIC = (3+1)*(3+1)*(3+1);
  constexpr int NPE_QUADRATIC = (2+1)*(2+1)*(2+1);

  std::vector<dray::int32> offsets(NPE_CUBIC);
  std::iota(offsets.begin(), offsets.end(), 0);

  std::vector<Vec<Float, WORLD_COMP>> displacement(NPE_QUADRATIC, {{1.0, 1.0, 1.0}});
  std::vector<Vec<Float, SCALAR_COMP>> scalar_value(NPE_CUBIC, {{1.0}});

  // Dummy element attributes for compiling only.
  const dray::ShapeHex melem_shape;
  const dray::OrderPolicy<dray::General> melem_order_p{2};
  const dray::ReadDofPtr<Vec<Float, WORLD_COMP>>
      melem_rdp{&offsets[0], &displacement[0]};

  const dray::ShapeHex felem_shape;
  const dray::OrderPolicy<dray::General> felem_order_p{3};
  const dray::ReadDofPtr<Vec<Float, SCALAR_COMP>>
      felem_rdp{&offsets[0], &scalar_value[0]};

  const Vec<Float, REFERENCE_COMP> ref = {{0.5, 0.5, 0.5}};


  // Evaluate Phi and derivatives.
  Vec<Vec<Float, WORLD_COMP>, REFERENCE_COMP> J; // each vec is col, according to eval_d
  Vec<Float, WORLD_COMP> world = dray::eops::eval_d(melem_shape, melem_order_p, melem_rdp, ref, J);

  Matrix<Vec<Float, WORLD_COMP>, REFERENCE_COMP, REFERENCE_COMP> D2_Phi;
  dray::eops::eval_hessian(melem_shape, melem_order_p, melem_rdp, ref, D2_Phi);

  // Get the inverse-transpose of the Jacobian.
  bool inv_valid;
  //  du   |  dv   |  dw     dudvdw  (dudvdw)^t
  // 1 2 3 | 4 5 6 | 7 8 9 -> 1 4 7 -> 1 2 3
  //                          2 5 8 -> 4 5 6
  //                          3 6 9 -> 7 8 9
  Matrix<Float, REFERENCE_COMP, WORLD_COMP> Jt;
  Jt.transpose_from_cols(J);
  dray::MatrixInverse<Float, REFERENCE_COMP> Jt_inv(Jt, inv_valid);  // LU decomposition
  // REFERENCE_COMP == WORLD_COMP == 3.

  // Evaluate f (scalar field) and derivatives w.r.t. reference space.
  Vec<Vec<Float, SCALAR_COMP>, REFERENCE_COMP> grad_f_ref;
  /// Vec<Float, 1> f = field_elem.eval_d(ref, grad_f_ref);
  Vec<Float, SCALAR_COMP> f = dray::eops::eval_d(felem_shape, felem_order_p, felem_rdp, ref, grad_f_ref);
  Vec<Float, REFERENCE_COMP> D1_f_ref = dray::squeeze(grad_f_ref); // ->  Vec<vec<1>> -> Vec

  Matrix<Vec<Float, SCALAR_COMP>, REFERENCE_COMP, REFERENCE_COMP> D2_f_ref_nested;
  dray::eops::eval_hessian(felem_shape, felem_order_p, felem_rdp, ref, D2_f_ref_nested);
  Matrix<Float, REFERENCE_COMP, REFERENCE_COMP> D2_f_ref;
  for (int i = 0; i < REFERENCE_COMP; ++i)
    for (int j = 0; j < REFERENCE_COMP; ++j)
      D2_f_ref(i,j) = D2_f_ref_nested(i,j)[0];

  // 1st derivative of f (scalar field) w.r.t. world space.
  Vec<Float, WORLD_COMP> D1_f_world = Jt_inv * D1_f_ref;


  // ===========================================================================
  // The formula
  // ===========================================================================
  const Float grad_mag = D1_f_world.magnitude();

  Matrix<Float, REFERENCE_COMP, REFERENCE_COMP> D2_Phi_dot_D1_f_world;
  for (int i = 0; i < REFERENCE_COMP; ++i)
    for (int j = 0; j < REFERENCE_COMP; ++j)
      D2_Phi_dot_D1_f_world(i,j) = dot(D2_Phi(i,j), D1_f_world);

  const Vec<Float, WORLD_COMP> D1_grad_mag_world =
      Jt_inv * ( (D2_f_ref - D2_Phi_dot_D1_f_world) * D1_f_world.normalized() );
  //
  //              (mat     -         mat            ) *  vec
  //   -------    -------------------------------------------------------------
  //   mat inv  *            vec
  //
  // ===========================================================================
}





TEST(dray_hessian, dray_vec_mag_grad)
{
  constexpr int SCALAR_COMP = 1;
  constexpr int VECTOR_COMP = 3;

  constexpr int NPE_CUBIC = (3+1)*(3+1)*(3+1);

  std::vector<dray::int32> offsets(NPE_CUBIC);
  std::iota(offsets.begin(), offsets.end(), 0);

  std::vector<dray::Vec<dray::Float, VECTOR_COMP>> vecs(NPE_CUBIC, {{1.0, 1.0, 1.0}});

  // Dummy element attributes for compiling only.
  const dray::ShapeHex melem_shape;
  const dray::OrderPolicy<dray::General> melem_order_p{2};
  const dray::ReadDofPtr<dray::Vec<dray::Float, VECTOR_COMP>>
      melem_rdp{&offsets[0], &vecs[0]};

  const dray::ShapeHex felem_shape;
  const dray::OrderPolicy<dray::General> felem_order_p{3};
  const dray::ReadDofPtr<dray::Vec<dray::Float, VECTOR_COMP>>
      felem_rdp{&offsets[0], &vecs[0]};

  const dray::Vec<dray::Float, 3> ref = {{0.5, 0.5, 0.5}};

  // Evaluate Phi and Jacobian.
  dray::Vec<dray::Vec<dray::Float, 3>, 3> J;
  dray::Vec<dray::Float, 3> world = dray::eops::eval_d(melem_shape, melem_order_p, melem_rdp, ref, J);

  // Get the inverse-transpose of the Jacobian.
  bool inv_valid;
  dray::Matrix<dray::Float, 3, 3> Jt;
  Jt.transpose_from_cols(J);
  dray::MatrixInverse<dray::Float, 3> Jt_inv(Jt, inv_valid);  // LU decomposition

  // Evaluate v (vector field) and derivative w.r.t. reference space.
  dray::Vec<dray::Vec<dray::Float, 3>, 3> D1_v_ref;
  dray::Vec<dray::Float, 3> v = dray::eops::eval_d(felem_shape, felem_order_p, felem_rdp, ref, D1_v_ref);

  dray::Matrix<dray::Float, 3, 3> D1_v_ref_t;
  D1_v_ref_t.transpose_from_cols(D1_v_ref);

  // Derivative of v (vector field) w.r.t. world space.
  dray::Matrix<dray::Float, 3, 3> D1_v_world_t = Jt_inv * D1_v_ref_t;

  // ===========================================================================
  // Gradient of vector magnitude.
  // ===========================================================================
  dray::Vec<dray::Float, 3> vec_mag_grad = D1_v_world_t * v.normalized();
  // ===========================================================================
}

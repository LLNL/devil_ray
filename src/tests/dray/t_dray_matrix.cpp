#include "gtest/gtest.h"
#include <dray/matrix.hpp>

TEST(dray_array, dray_matrix_inverse)
{
  dray::Matrix<float,3,3> mat;
  mat[0][0] = 1;
  mat[1][0] = 2;
  mat[2][0] = 3;

  mat[0][1] = 0;
  mat[1][1] = 1;
  mat[2][1] = 4;

  mat[0][2] = 5;
  mat[1][2] = 6;
  mat[2][2] = 0;

  std::cout<<mat; 

  bool valid;
  dray::Matrix<float,3,3> inv = dray::matrix_inverse(mat, valid);
  std::cout<<inv;
  float abs_error  = 0.0001f;

  ASSERT_NEAR(inv[0][0], -24.f, abs_error);
  ASSERT_NEAR(inv[1][0], 20.f, abs_error);
  ASSERT_NEAR(inv[2][0], -5.f, abs_error);
                       
  ASSERT_NEAR(inv[0][1], 18.f, abs_error);
  ASSERT_NEAR(inv[1][1], -15.f, abs_error);
  ASSERT_NEAR(inv[2][1], 4.f, abs_error);
                       
  ASSERT_NEAR(inv[0][2], 5.f, abs_error);
  ASSERT_NEAR(inv[1][2], -4.f, abs_error);
  ASSERT_NEAR(inv[2][2], 1.f, abs_error);
  
}

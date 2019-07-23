#include "gtest/gtest.h"
#include "test_config.h"
#include "t_utils.hpp"

#include <stdio.h>

#include <dray/utils/png_encoder.hpp>
#include <dray/filters/surface_triangle.hpp>
#include <dray/Element/element.hpp>
#include <dray/Element/pos_simplex_element.hpp>

const int c_width = 1024;
const int c_height = 1024;
const int num_samples = 500000;

TEST(dray_triangle, dray_triangle_single)
{
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "triangle");
  remove_test_image(output_file);

  using Coord = dray::Vec<float,2>;
  using Color = dray::Vec<float,4>;
  dray::Array<Color> img_buffer;

  // Define a linear triangle.
  Coord linear_triangle[3] = { {0.0f, 0.0f},
                               {.75f, 0.0f},
                               {0.0f, .75f},
  };

  // Define a flat quadratic triangle.
  Coord quadratic_triangle[6] = { {0.0f, 0.0f},  {0.4f, 0.0f},  {0.8f, 0.0f},
                                  {0.0f, 0.4f},  {0.4f, 0.4f},
                                  {0.0f, 0.8f},
  };

  // Define a curvy quadratic triangle.
  const float curvy = 1.0f;
  const float sqt3 = pow(3.0f, 0.5);
  Coord curvy_quadratic_triangle[6] = { {0.0f, 0.0f},  {0.5f, 0.0f},  {1.0f, 0.0f},
                                               {.25f, sqt3/4},  {.75f, sqt3/4},
                                                       {0.5f, sqt3/2},
  };
  const Coord centroid = {0.5f, 0.5f/sqt3};
  curvy_quadratic_triangle[1] = curvy_quadratic_triangle[1] * (1-curvy)  +  centroid * curvy;
  curvy_quadratic_triangle[3] = curvy_quadratic_triangle[3] * (1-curvy)  +  centroid * curvy;
  curvy_quadratic_triangle[4] = curvy_quadratic_triangle[4] * (1-curvy)  +  centroid * curvy;

  /// dray::Array<Coord> nodes_array(linear_triangle, 3);
  /// const int poly_order = 1;
  /// dray::Array<Coord> nodes_array(quadratic_triangle, 6);
  /// const int poly_order = 2;
  dray::Array<Coord> nodes_array(curvy_quadratic_triangle, 6);
  const int poly_order = 2;

  img_buffer = dray::SurfaceTriangle().execute<float>(c_width, c_height, nodes_array, poly_order, num_samples);

  // Save image.
  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) img_buffer.get_host_ptr(), c_width, c_height);
  png_encoder.save(output_file + ".png");

  // Check against benchmark.
  EXPECT_TRUE(check_test_image(output_file));
}


TEST(dray_triangle, dray_tetrahedron_single)
{
  using T = float;
  using DofT = dray::Vec<T,3>;

  constexpr auto Tri = dray::newelement::ElemType::Tri;
  constexpr auto GeneralOrder = dray::newelement::Order::General;

  const int p = 5;

  dray::Vec<T, 3> identity_dofs[(p+1)*(p+2)*(p+3)/6];
  int dof_idx = 0;

  for (int kk = 0; kk <= p; kk++)
  {
    for (int jj = 0; jj <= p-kk; jj++)
    {
      for (int ii = 0; ii <= p-kk-jj; ii++)
      {
        identity_dofs[dof_idx++] = {T(ii)/p, T(jj)/p, T(kk)/p};
      }
    }
  }

  dray::newelement::Element<T, 3, Tri, GeneralOrder> my_tetrahedron;
  my_tetrahedron.construct(p);

  for (int kk = 0; kk <= p; kk++)
  {
    for (int jj = 0; jj <= p-kk; jj++)
    {
      for (int ii = 0; ii <= p-kk-jj; ii++)
      {
        dray::Vec<T,3> ref = {T(ii)/p, T(jj)/p, T(kk)/p};
        dray::Vec<T,3> loc = my_tetrahedron.eval<DofT>(ref, identity_dofs);
        EXPECT_FLOAT_EQ(ref[0], loc[0]);
        EXPECT_FLOAT_EQ(ref[1], loc[1]);
        EXPECT_FLOAT_EQ(ref[2], loc[2]);
      }
    }
  }

}

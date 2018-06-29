#include "gtest/gtest.h"
#include <dray/high_order_shape.hpp>
#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/math.hpp>
#include <dray/binomial.hpp>
#include <dray/utils/png_encoder.hpp>

#include <iostream>
#include <string>
#include <string.h>


dray::Vec4f color_2d_grid(float ref1, float ref2)
{
  constexpr int ch1 = 0;  // ref1 -> red
  constexpr int ch2 = 2;  // ref2 -> blue
  constexpr int chb = 1;  // background -> green

  dray::Vec4f color = {0, 0, 0, 1};
  color[ch1] = ref1;
  color[ch2] = ref2;
  color[chb] = .5*sqrt(1 - pow(.5f*ref1 + .5f*ref2, 2));  // Some normalization.

  return color;
}

int max_idx(int arr_size, const float *arr, int stride)
{
  int max_idx = 0;
  for (int ii = 1; ii < arr_size; ii++)
    if (arr[ii*stride] > arr[max_idx*stride])
      max_idx = ii;
  return max_idx;
}

void make_picture(const float *weights, int stride, int el_dofs_1d, int img_w, const std::string &filename)
{
  const int img_size = img_w * img_w;

  dray::Array<dray::Vec4f> img_buffer;
  img_buffer.resize(img_size);

  const int el_dofs = el_dofs_1d * el_dofs_1d;

  // Color the image.
  dray::Vec4f *img_ptr = img_buffer.get_host_ptr();
  for (int ii = 0; ii < img_size; ii++)
  {
    // Determine the color by node logical position.
    // Note that bshape orders the nodes by iterating first over y, then over x.
    const int brightest_idx = max_idx(el_dofs, weights + ii*el_dofs*stride, stride);
    const float node_x = (brightest_idx / el_dofs_1d) / static_cast<float>(el_dofs_1d);
    const float node_y = (brightest_idx % el_dofs_1d) / static_cast<float>(el_dofs_1d);

    img_ptr[ii] = color_2d_grid(node_x, node_y);
  }

  // Output image.
  dray::PNGEncoder png_encoder;
  png_encoder.encode((float*) img_ptr, img_w, img_w);
  png_encoder.save(filename);
}

void visualize_bernstein2d(int order, int _img_w = 100)
{
  const int img_w = _img_w;
  const int img_size = img_w * img_w;

  dray::Array<int> active_idx;
  dray::Array<dray::Vec<float,2>> ref_pts;
  active_idx.resize(img_size);
  ref_pts.resize(img_size);

  // Initialize active_idx and ref_pts.
  int *active_idx_ptr = active_idx.get_host_ptr();
  dray::Vec<float,2> *ref_pts_ptr = ref_pts.get_host_ptr();
  for (int ii = 0; ii < img_size; ii++)
  {
     active_idx_ptr[ii] = ii;

     // Pixel coordinates, left to right, top to bottom.
     // Center of pixel.
    const int px_x = ii % img_w;
    const int px_y = ii / img_w;
    ref_pts_ptr[ii] = { (px_x + 0.5f) / img_w, (px_y + 0.5f) / img_w };
  }

  // Evaluate the shape function.
  dray::BernsteinShape<float,2> bshape;
  bshape.m_p_order = order;
  dray::Array<float> shape_val;
  dray::Array<dray::Vec<float,2>> shape_deriv;
  bshape.calc_shape_dshape(active_idx, ref_pts, shape_val, shape_deriv);

  //const int el_dofs = bshape.get_el_dofs();
  const int el_dofs_1d = bshape.get_el_dofs_1d();

  // Output visualization for value, x-derivative, and y-derivative.
  make_picture(shape_val.get_host_ptr_const(), 1, el_dofs_1d, img_w, "bernstein_vals.png");
  make_picture((float*) shape_deriv.get_host_ptr_const(), 2, el_dofs_1d, img_w, "bernstein_xderiv.png");
  make_picture((float*) shape_deriv.get_host_ptr_const() + 1, 2, el_dofs_1d, img_w, "bernstein_yderiv.png");
}




TEST(dray_test, dray_high_order_shape)
{

  visualize_bernstein2d(8, 500);

  constexpr int DOF = dray::IntPow<2+1,3>::val;

  //--- Test classes ---//
  ///dray::BernsteinShape<float, 2> bshape = {4};
  ///std::cout << "bshape el_dofs = " << bshape.get_el_dofs() << std::endl;


  //--- Test linear shape evaluator---//    Linear Shape works.
  {
    /// // Make 3 queries on a trilinear element.
    /// int _active_idx[3] = {0, 1, 2};
    /// dray::Array<int> active_idx(_active_idx, 3);
    /// dray::Vec<float,3> _ref_pts[3] = {{0.,0.,0.}, {.5,.5,.5}, {.25,.75,1.}};
    /// dray::Array<dray::Vec<float,3>> ref_pts(_ref_pts, 3);
  
    /// dray::LinearShape<float, 3> lshape;
    /// dray::Array<float> shape_val;
    /// dray::Array<dray::Vec<float,3>> shape_deriv;
    /// lshape.calc_shape_dshape(active_idx, ref_pts, shape_val, shape_deriv);
  
    /// std::cout << "LinearShape" << std::endl;
    /// std::cout << "ref_pts   ";   ref_pts.summary();  std::cout << std::endl;
    /// std::cout << "shape_val   ";   shape_val.summary();  std::cout << std::endl;
    /// std::cout << "shape_deriv   ";   shape_deriv.summary();  std::cout << std::endl;
  }

  //--- Test Bernstein shape evaluator---//
  {
    constexpr int RefDim = 2;
    // Make 3 queries on a 1D Bernstein element.
    int _active_idx[3] = {0, 1, 2};
    dray::Array<int> active_idx(_active_idx, 3);
    //dray::float32 _ref_pts[3] = {0., .5, .25};
    dray::Vec<float,2> _ref_pts[3] = {{0.,0.}, {.5,.5}, {.25,.75}};
    ///dray::Vec<float,3> _ref_pts[3] = {{0.,0.,0.}, {.5,.5,.5}, {.25,.75,1.}};
    dray::Array<dray::Vec<float,RefDim>> ref_pts(_ref_pts, 3);
  
    dray::BernsteinShape<float, RefDim> bshape;
    bshape.m_p_order = 2;
    dray::Array<float> shape_val;
    //dray::Array<dray::Vec<float,3>> shape_deriv;
    dray::Array<dray::Vec<float,RefDim>> shape_deriv;
    bshape.calc_shape_dshape(active_idx, ref_pts, shape_val, shape_deriv);

    std::cout << "BernsteinShape" << std::endl;
    std::cout << "eldofs == " << bshape.get_el_dofs() << std::endl;
    std::cout << "ref_pts   ";   ref_pts.summary();  std::cout << std::endl;
    std::cout << "shape_val   ";   shape_val.summary();  std::cout << std::endl;
    std::cout << "shape_deriv   ";   shape_deriv.summary();  std::cout << std::endl;
  }


  //--- Test binomial coefficients <TMP>---//    //This works
  /// std::cout << dray::BinomT<5,0>::val << " "
  ///           << dray::BinomT<5,1>::val << " "
  ///           << dray::BinomT<5,2>::val << " "
  ///           << dray::BinomT<5,3>::val << " "
  ///           << dray::BinomT<5,4>::val << " "
  ///           << dray::BinomT<5,5>::val << std::endl;


  /// const float *binom_row = dray::BinomRowT<float, 8>::get_static();
  /// std::cout << binom_row[0] << " "
  ///           << binom_row[1] << " "
  ///           << binom_row[2] << " "
  ///           << binom_row[3] << " "
  ///           << binom_row[4] << " "
  ///           << binom_row[5] << " "
  ///           << binom_row[6] << " "
  ///           << binom_row[7] << " "
  ///           << binom_row[8] << std::endl;

  /// dray::BinomRowT<float, 6> binom_row_obj;
  /// binom_row = binom_row_obj.get();
  /// std::cout << binom_row[0] << " "
  ///           << binom_row[1] << " "
  ///           << binom_row[2] << " "
  ///           << binom_row[3] << " "
  ///           << binom_row[4] << " "
  ///           << binom_row[5] << " "
  ///           << binom_row[6] << std::endl;


  //--- Test binomial coefficients from BinomTable---/   //This works
  {
    /// dray::GlobBinomTable.size_at_least(7);
    /// dray::GlobBinomTable.m_rows.summary();

    /// int row_idx = 5;
    /// const int *table_ptr = dray::GlobBinomTable.get_host_ptr_const();
    /// const int *row_ptr = dray::BinomTable::get_row(table_ptr, row_idx);
    /// for (int kk = 0; kk <= row_idx; kk++)
    /// {
    ///   std::cout << row_ptr[kk] << " ";
    /// }
    /// std::cout << std::endl;

    /// int single_row[6+1];
    /// dray::BinomRow<int>::fill_single_row(6, single_row);
    /// std::cout << "single row" << std::endl;
    /// for (int ii = 0; ii <= 6; ii++)
    /// {
    ///   std::cout << single_row[ii] << " ";
    /// }
    /// std::cout << std::endl;
  }

  //--- Test ElTrans ---//
  {
    dray::BernsteinShape<float, 3> bshape;
    bshape.m_p_order = 2;
    dray::ElTrans_BernsteinShape<float,1,3>  eltrans;
    eltrans.resize(2, 27, bshape, 45);
    
    // There are two quadratic unit-cubes, adjacent along X, sharing a face in the YZ plane.
    // There are 45 total control points: 2 vol mids, 11 face mids, 20 edge mids, and 12 vertices.
    float grid_vals[45] = 
        { 10, -10,                           // 0..1 vol mids A and B
          15,7,7,7,7,  0, -15,-7,-7,-7,-7,      // 2..12 face mids A(+X,+Y,+Z,-Y,-Z) AB B(-X,+Y,+Z,-Y,-Z)
          12,12,12,12,  -12,-12,-12,-12,     // 13..20 edge mids on ends +X/-X A(+Y,+Z,-Y,-Z) B(+Y,+Z,-Y,-Z)
          5,5,5,5,  -5,-5,-5,-5,             // 21..28 edge mids YZ corners  A(++,-+,--,+-) B(++,-+,--,+-)
          0,0,0,0,                           // 29..32 edge mids on shared face AB(+Y,+Z,-Y,-Z)
          20,20,20,20,  -20,-20,-20,-20,     // 33..40 vertices on ends +X/-X, YZ corners A(++,-+,--,+-) B(++,-+,--,+-)
          0,0,0,0 };                         // 41..44 vertices on shared face, YZ corners AB(++,-+,--,+-)

    // Map the per-element degrees of freedom into the total set of control points.
    int ctrl_idx[54];
    int * const ax = ctrl_idx, * const bx = ctrl_idx + 27;

    // Nonshared nodes.
    ax[13] = 0;  bx[13] = 1;

    ax[22] = 2;  bx[4] = 8;
    ax[16] = 3;  bx[16] = 9;
    ax[14] = 4;  bx[14] = 10;
    ax[10] = 5;  bx[10] = 11;
    ax[12] = 6;  bx[12] = 12;

    ax[25] = 13; bx[7] = 17;
    ax[23] = 14; bx[5] = 18;
    ax[19] = 15; bx[1] = 19;
    ax[21] = 16; bx[3] = 20;

    ax[17] = 21; bx[17] = 25;
    ax[11] = 22; bx[11] = 26;
    ax[9] = 23;  bx[9] = 27;
    ax[15] = 24; bx[15] = 28;

    ax[26] = 33; bx[8] = 37;
    ax[20] = 34; bx[2] = 38;
    ax[18] = 35; bx[0] = 39;
    ax[24] = 36; bx[6] = 40;

    // Shared nodes.
    ax[4]    =   bx[22] = 7;
    
    ax[7]    =   bx[25] = 29;
    ax[5]    =   bx[23] = 30;
    ax[1]    =   bx[19] = 31;
    ax[3]    =   bx[21] = 32;

    ax[8]    =   bx[26] = 41;
    ax[2]    =   bx[20] = 42;
    ax[0]    =   bx[18] = 43;
    ax[6]    =   bx[24] = 44;

    // Initialize eltrans with these values.
    memcpy( eltrans.get_m_ctrl_idx().get_host_ptr(), ctrl_idx, 54*sizeof(int) );
    memcpy( eltrans.get_m_values().get_host_ptr(), grid_vals, 45*sizeof(float) );

    // Evaluate.
    constexpr int num_queries = 4;
    int _active_idx[num_queries] = {0,1, 2,3};
    dray::Array<int> active_idx(_active_idx, num_queries);

    int _el_ids[num_queries] = {0,0, 1,1};
    dray::Array<int> el_ids(_el_ids, num_queries);

    float _ref_pts[3*num_queries] =
        { .5,.9,.9,
          .9,.9,.9,
          .5,.9,.9,
          .1,.9,.9 };
    dray::Array<dray::Vec<float,3>> ref_pts( (dray::Vec<float,3> *) _ref_pts, num_queries);

    dray::Array<dray::Vec<float,1>> trans_val;
    dray::Array<dray::Matrix<float,1,3>>  trans_deriv;
    trans_val.resize(num_queries);
    trans_deriv.resize(num_queries);

    eltrans.eval(active_idx, el_ids, ref_pts, trans_val, trans_deriv);
    
    std::cout << "Test ElTrans" << std::endl;
    std::cout << "active_idx ";  active_idx.summary();
    std::cout << "el_ids ";      el_ids.summary();
    std::cout << "ref_pts ";     ref_pts.summary();
    std::cout << "trans_val ";   trans_val.summary();
    std::cout << "trans_deriv "; trans_deriv.summary();
    std::cout << std::endl;
  }

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
  /// dray::Array<dray::Vec<float, num_field_comp>> elt_vals;
  /// elt_vals = scalar_field.eval(DummyFunctor(), ref_pts);

  /// // Report
  /// elt_vals.summary();
}

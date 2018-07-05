#include "gtest/gtest.h"

#include <dray/high_order_shape.hpp>


TEST(dray_test, dray_newton_solve)
{
  // Test NewtonSolve
  {
    // For this test we will use the R3->R3 transformation of {ref space} -> {phys space}.
    typedef dray::BernsteinShape<float,3> ShType;
    typedef dray::ElTrans_BernsteinShape<float,3,3> ElTType;
    typedef dray::ElTransQuery<ElTType> QType;
    typedef dray::NewtonSolve<QType> NSType;

    // There are two quadratic unit-cubes, adjacent along X, sharing a face in the YZ plane.
    // There are 45 total control points: 2 vol mids, 11 face mids, 20 edge mids, and 12 vertices.

    ShType bshape;
    bshape.m_p_order = 2;
    QType field_query;
    
        // 2 elts, 27 el_dofs, supply instance of ShType, 45 total control points.
    field_query.m_eltrans.resize(2, 27, bshape, 45);

    // Scalar field values of control points.
    /// float grid_vals[45] = 
    ///     { 10, -10,                           // 0..1 vol mids A and B
    ///       15,7,7,7,7,  0, -15,-7,-7,-7,-7,   // 2..12 face mids A(+X,+Y,+Z,-Y,-Z) AB B(-X,+Y,+Z,-Y,-Z)
    ///       12,12,12,12,  -12,-12,-12,-12,     // 13..20 edge mids on ends +X/-X A(+Y,+Z,-Y,-Z) B(+Y,+Z,-Y,-Z)
    ///       5,5,5,5,  -5,-5,-5,-5,             // 21..28 edge mids YZ corners  A(++,-+,--,+-) B(++,-+,--,+-)
    ///       0,0,0,0,                           // 29..32 edge mids on shared face AB(+Y,+Z,-Y,-Z)
    ///       20,20,20,20,  -20,-20,-20,-20,     // 33..40 vertices on ends +X/-X, YZ corners A(++,-+,--,+-) B(++,-+,--,+-)
    ///       0,0,0,0 };                         // 41..44 vertices on shared face, YZ corners AB(++,-+,--,+-)

    // Physical space locations of control points. (Non-deformed cubes).
    float grid_loc[3*45] =
           { .5, .5, .5,    // 0
            -.5, .5, .5,    // 1

              1, .5, .5,    // 2
             .5,  1, .5,    // 3
             .5, .5,  1,    // 4
             .5,  0, .5,    // 5
             .5, .5,  0,    // 6

              0, .5, .5,    // 7

             -1, .5, .5,    // 8
            -.5,  1, .5,    // 9
            -.5, .5,  1,    // 10
            -.5,  0, .5,    // 11
            -.5, .5,  0,    // 12

              1,  1, .5,    // 13
              1, .5,  1,    // 14
              1,  0, .5,    // 15
              1, .5,  0,    // 16

             -1,  1, .5,    // 17
             -1, .5,  1,    // 18
             -1,  0, .5,    // 19
             -1, .5,  0,    // 20

             .5,  1,  1,    // 21
             .5,  0,  1,    // 22
             .5,  0,  0,    // 23
             .5,  1,  0,    // 24

            -.5,  1,  1,    // 25
            -.5,  0,  1,    // 26
            -.5,  0,  0,    // 27
            -.5,  1,  0,    // 28

              0,  1, .5,    // 29
              0, .5,  1,    // 30
              0,  0, .5,    // 31
              0, .5,  0,    // 32

              1,  1,  1,    // 33
              1,  0,  1,    // 34
              1,  0,  0,    // 35
              1,  1,  0,    // 36

             -1,  1,  1,    // 37
             -1,  0,  1,    // 38
             -1,  0,  0,    // 39
             -1,  1,  0,    // 40

              0,  1,  1,    // 41
              0,  0,  1,    // 42
              0,  0,  0,    // 43
              0,  1,  0  }; // 44

              

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
    memcpy( field_query.m_eltrans.get_m_ctrl_idx().get_host_ptr(), ctrl_idx, 54*sizeof(int) );
    //memcpy( field_query.m_eltrans.get_m_values().get_host_ptr(), grid_vals, 45*sizeof(float) );  //scalar field values
    memcpy( field_query.m_eltrans.get_m_values().get_host_ptr(), grid_loc, 3*45*sizeof(float) );  //space locations

    // Set up query.
    constexpr int num_queries = 4;
    field_query.resize(num_queries);

    int _el_ids[num_queries] = {0,0, 1,1};
    dray::Array<int> el_ids(_el_ids, num_queries);

    // The target points.
    float _tgt_pts[3*num_queries] =
        { .5,.9,.9,
          .9,.9,.9,
         -.5,.9,.9,
         -.9,.9,.9 };
    dray::Array<dray::Vec<float,3>> tgt_pts( (dray::Vec<float,3> *) _tgt_pts, num_queries);

    //// // Really good initial guesses.
    //// float _ref_pts[3*num_queries] =
    ////     { .5,.9,.9,
    ////       .9,.9,.9,
    ////       .5,.9,.9,
    ////       .1,.9,.9 };
    //// dray::Array<dray::Vec<float,3>> ref_pts( (dray::Vec<float,3> *) _ref_pts, num_queries);

    // Centered initial guesses.
    float _ref_pts[3*num_queries] =
        { .5,.5,.5,
          .5,.5,.5,
          .5,.5,.5,
          .5,.5,.5 };
    dray::Array<dray::Vec<float,3>> ref_pts( (dray::Vec<float,3> *) _ref_pts, num_queries);

    field_query.m_el_ids = el_ids;
    field_query.m_ref_pts = ref_pts;

    // Output init states.
    std::cout << "Test NewtonSolve."  << std::endl;
    std::cout << "Target points:   "; tgt_pts.summary();
    std::cout << "Element ids:     "; el_ids.summary();
    std::cout << "Init guesses:    "; ref_pts.summary();

    int _active_idx[num_queries] = {0,1, 2,3};
    dray::Array<int> active_idx(_active_idx, num_queries);

    // Perform the solve.
    dray::Array<int> solve_status;
    int num_iterations = NSType::step(tgt_pts, field_query, active_idx, 10, solve_status);

    // Output results.
    std::cout << "Num iterations:  " << num_iterations << std::endl;
    std::cout << "Solve status:    "; solve_status.summary();
    std::cout << "Final ref pts:   "; field_query.m_ref_pts.summary();
    std::cout << "Final phys pts:  "; field_query.m_result_val.summary();
    std::cout << std::endl;

  }  // scoping

}

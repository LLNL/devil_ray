#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/high_order_shape.hpp>
#include <dray/newton_solver.hpp>

#include <dray/camera.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/math.hpp>


void print_rays(dray::Array<dray::Ray<float>> rays)
{
  printf("rays.m_hit_idx...\n");
  const dray::Ray<float> *ray_ptr = rays.get_host_ptr_const();
  for (int i = 0; i < rays.size(); ++i)
  {
    printf("%d ", ray_ptr[i].m_hit_idx);
  }
  printf("\n");
}



TEST(dray_test, dray_newton_solve)
{
  // Set up the mesh / field.

// For this test we will use the R3->R3 transformation of {ref space} -> {phys space}.

  // There are two quadratic unit-cubes, adjacent along X, sharing a face in the YZ plane.
  // There are 45 total control points: 2 vol mids, 11 face mids, 20 edge mids, and 12 vertices.

  // 2 elts, 27 el_dofs, supply instance of ShType, 45 total control points.
  dray::ElTransData<float,3> eltrans_space;
  dray::ElTransData<float,1> eltrans_field;
  eltrans_space.resize(2, 27, 45);
  eltrans_field.resize(2, 27, 45);

  // Scalar field values of control points.
  float grid_vals[45] =
      { 10, -10,                           // 0..1 vol mids A and B
        15,7,7,7,7,  0, -15,-7,-7,-7,-7,   // 2..12 face mids A(+X,+Y,+Z,-Y,-Z) AB B(-X,+Y,+Z,-Y,-Z)
        12,12,12,12,  -12,-12,-12,-12,     // 13..20 edge mids on ends +X/-X A(+Y,+Z,-Y,-Z) B(+Y,+Z,-Y,-Z)
        5,5,5,5,  -5,-5,-5,-5,             // 21..28 edge mids YZ corners  A(++,-+,--,+-) B(++,-+,--,+-)
        0,0,0,0,                           // 29..32 edge mids on shared face AB(+Y,+Z,-Y,-Z)
        20,20,20,20,  -20,-20,-20,-20,     // 33..40 vertices on ends +X/-X, YZ corners A(++,-+,--,+-) B(++,-+,--,+-)
        0,0,0,0 };                         // 41..44 vertices on shared face, YZ corners AB(++,-+,--,+-)

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

// Initialize eltrans space and field with these values.
memcpy( eltrans_field.m_ctrl_idx.get_host_ptr(), ctrl_idx, 54*sizeof(int) );
memcpy( eltrans_space.m_ctrl_idx.get_host_ptr(), ctrl_idx, 54*sizeof(int) );
memcpy( eltrans_field.m_values.get_host_ptr(), grid_vals, 45*sizeof(float) );   //scalar field values
memcpy( eltrans_space.m_values.get_host_ptr(), grid_loc, 3*45*sizeof(float) );  //space locations

// Test NewtonSolve
{
  constexpr int num_queries = 4;

  int _el_ids[num_queries] = {0,0, 1,1};

  // The target points.
  float _tgt_pts[3*num_queries] =
      { .5,.9,.9,
        .9,.9,.9,
       -.5,.9,.9,
       -.9,.9,.9 };
  dray::Vec<float,3> *tgt_pts = (dray::Vec<float,3> *) _tgt_pts;

  //// // Really good initial guesses.
  //// float _ref_pts[3*num_queries] =
  ////     { .5,.9,.9,
  ////       .9,.9,.9,
  ////       .5,.9,.9,
  ////       .1,.9,.9 };
  //// <dray::Vec<float,3> *ref_pts = (dray::Vec<float,3> *) _ref_pts;

  // Centered initial guesses.
  float _ref_pts[3*num_queries] =
      { .5,.5,.5,
        .5,.5,.5,
        .5,.5,.5,
        .5,.5,.5 };
  dray::Vec<float,3> *ref_pts = (dray::Vec<float,3> *) _ref_pts;

  // Output init states.
  std::cout << "Test NewtonSolve."  << std::endl;
  std::cout << "Target points:   "; for(int ii=0; ii<num_queries; ii++) printf("(%f %f %f)  ", tgt_pts[ii][0], tgt_pts[ii][1], tgt_pts[ii][2]); printf("\n");
  std::cout << "Element ids:     "; for(int ii=0; ii<num_queries; ii++) printf("(%d)  ", _el_ids[ii]); printf("\n");
  std::cout << "Init guesses:    "; for(int ii=0; ii<num_queries; ii++) printf("(%f %f %f)  ", ref_pts[ii][0], ref_pts[ii][1], ref_pts[ii][2]); printf("\n");

  int num_iterations[num_queries];
  int solve_status[num_queries];

  typedef dray::BernsteinBasis<float,3>                                    ShapeOpType;
  typedef dray::ElTransOp<float, ShapeOpType, dray::ElTransIter<float,3> > TransOpType;

  // Auxiliary memory.
  float aux_array[2 * 6*3];

  // Perform the solve.
  for (int qidx = 0; qidx < num_queries; qidx++)
  {
    TransOpType trans;
    trans.init_shape(2);///, aux_array);
    trans.m_coeff_iter.init_iter(eltrans_space.m_ctrl_idx.get_host_ptr(),
                                 eltrans_space.m_values.get_host_ptr(),
                                 27, _el_ids[qidx]);

    solve_status[qidx] = dray::NewtonSolve<float>::solve<TransOpType>(
        trans, tgt_pts[qidx], ref_pts[qidx], 0.00001, 0.00001, num_iterations[qidx]);
  }

  // Output results.
  std::cout << "Num iterations:  "; for(int ii=0; ii<num_queries; ii++) printf("(%d)  ", num_iterations[ii]); printf("\n");
  std::cout << "Solve status:    "; for(int ii=0; ii<num_queries; ii++) printf("(%d)  ", solve_status[ii]); printf("\n");
  std::cout << "Final ref pts:   "; for(int ii=0; ii<num_queries; ii++) printf("(%f %f %f)  ", ref_pts[ii][0], ref_pts[ii][1], ref_pts[ii][2]); printf("\n");
  std::cout << std::endl;

}  // Test NewtonSolve using a handful of points.

{

  dray::Mesh<float> mesh(eltrans_space, 2);
  dray::Field<float> field(eltrans_field, 2);
  dray::MeshField<float> mesh_field(mesh, field);

  constexpr int c_width = 1024;
  constexpr int c_height = 1024;

  //
  // Use camera to generate rays and points.
  //
  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.set_up(dray::make_vec3f(0,0,1));
  camera.set_pos(dray::make_vec3f(3.2,4.3,3));
  camera.set_look_at(dray::make_vec3f(0,0,0));
  //camera.reset_to_bounds(mesh_field.get_bounds());
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  /// //
  /// // Point location.
  /// //

  /// // For the single tips, use a fixed ray distance.
  /// for (int r = 0; r < rays.size(); r++)
  ///   rays.m_dist.get_host_ptr()[r] = 5.5;
  /// dray::Array<dray::Vec3f> points = rays.calc_tips();
  /// dray::int32 psize = points.size();

  ///     ////  const int psize = 100;
  ///     ////  const int mod = 1000000;
  ///     ////  dray::Array<dray::Vec3f> points;
  ///     ////  points.resize(psize);
  ///     ////  dray::Vec3f *points_ptr = points.get_host_ptr();

  ///     ////  // pick a bunch of random points inside the data bounds
  ///     ////  dray::AABB<> bounds = mesh_field.get_bounds();
  ///     ////  std::cout << "mesh_field bounds:  " << bounds << std::endl;

  ///     ////  float x_length = bounds.m_x.length();
  ///     ////  float y_length = bounds.m_y.length();
  ///     ////  float z_length = bounds.m_z.length();

  ///     ////  for(int i = 0;  i < psize; ++i)
  ///     ////  {
  ///     ////    float x = ((rand() % mod) / float(mod)) * x_length + bounds.m_x.min();
  ///     ////    float y = ((rand() % mod) / float(mod)) * y_length + bounds.m_y.min();
  ///     ////    float z = ((rand() % mod) / float(mod)) * z_length + bounds.m_z.min();

  ///     ////    points_ptr[i][0] = x;
  ///     ////    points_ptr[i][1] = y;
  ///     ////    points_ptr[i][2] = z;
  ///     ////  }

  ///   // active_rays: All are active.
  /// rays.m_active_rays.resize(rays.size());
  /// for (int r = 0; r < rays.size(); r++)
  ///   rays.m_active_rays.get_host_ptr()[r] = r;

  /// std::cout << "active_rays ||    ";
  /// rays.m_active_rays.summary();

  /// std::cout << "Test points (b locate):  ";
  /// points.summary();

  /// std::cout<<"locating\n";
  /// ///dray::Array<dray::int32> elt_ids;
  /// ///dray::Array<dray::Vec<float,3>> ref_pts;
  /// ///elt_ids.resize(psize);
  /// ///ref_pts.resize(psize);
  /// mesh_field.locate(points, rays.m_active_rays, rays.m_hit_idx, rays.m_hit_ref_pt);

  /// std::cerr << "Located, now summarizing." << std::endl;

  /// // Count how many have what element ids.
  /// constexpr int num_el = 2;
  /// int id_counts[num_el+1] = {0, 0, 0};  // There are two valid element ids. +1 for invalid.
  /// for (int ray_idx = 0; ray_idx < psize; ray_idx++)
  /// {
  ///   int hit_idx = rays.m_hit_idx.get_host_ptr_const()[ray_idx];
  ///   hit_idx = min( max( -1, hit_idx ), num_el );  // Clamp.
  ///   hit_idx = (hit_idx + num_el+1) % (num_el+1);
  ///   id_counts[hit_idx]++;
  ///   std::cout << "(" << ray_idx << ", " << hit_idx << ") ";
  /// }
  /// std::cout << std::endl;

  /// std::cout << "Test points (a locate):  ";
  /// points.summary();
  /// std::cout << "Element ids:  ";
  /// rays.m_hit_idx.summary();
  /// printf("(counts) [0]: %d  [1]: %d  [other]: %d\n", id_counts[0], id_counts[1], id_counts[2]);
  /// std::cout << "Ref pts:      ";
  /// rays.m_hit_ref_pt.summary();

  /// std::cerr << "Finished locating." << std::endl;

  /// //
  /// // Shading context.
  /// //
  /// dray::ShadingContext<dray::float32> shading_ctx = mesh_field.get_shading_context(rays);
  /// for (int aii = 0; aii < rays.m_active_rays.size(); aii++)
  ///   printf("%f  ",  shading_ctx.m_sample_val.get_host_ptr()[ rays.m_active_rays.get_host_ptr()[aii] ] );
  /// printf("\n");
  /// for (int aii = 0; aii < rays.m_active_rays.size(); aii++)
  /// {
  ///   int rii = rays.m_active_rays.get_host_ptr()[aii];
  ///   dray::Vec<float, 3> normal = shading_ctx.m_normal.get_host_ptr()[rii];
  ///   std::cout << normal << " ";
  /// }
  /// std::cout << std::endl;

  /// std::cerr << "Finished shading context." << std::endl;

  /// //
  /// // Volume rendering
  /// //
  /// float sample_dist = 0.01;
  /// dray::Array<dray::Vec<dray::float32,4>> color_buffer = mesh_field.integrate(rays, sample_dist);

  /// {
  /// dray::PNGEncoder png_encoder;
  /// png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
  /// png_encoder.save("volume_rendering.png");
  /// }


  /// //
  /// // Isosurface
  /// //

  /// print_rays(rays);

  /// mesh_field.intersect_isosurface(rays, 15.0);

  /// print_rays(rays);


  /// // Output rays to depth map.
  /// save_depth(rays, camera.get_width(), camera.get_height());

  std::string output_path = prepare_output_dir();
  // Output isosurface, colorized by field spatial gradient magnitude.
  {
    float isovalues[5] = { 15, 8, 0, -8, -15 };
    const char* filenames[5] = {"isosurface_+15",
                                "isosurface_+08",
                                "isosurface__00",
                                "isosurface_-08",
                                "isosurface_-15"};

    for (int iso_idx = 0; iso_idx < 5; iso_idx++)
    {
      std::string output_file = conduit::utils::join_file_path(output_path, std::string(filenames[iso_idx]));
      remove_test_image(output_file);

      dray::Array<dray::Vec4f> color_buffer = mesh_field.isosurface_gradient(rays, isovalues[iso_idx]);
      dray::PNGEncoder png_encoder;
      png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
      png_encoder.save(output_file + ".png");

      EXPECT_TRUE(check_test_image(output_file));
      printf("Finished rendering isosurface idx %d\n", iso_idx);
    }
  }


  /// // Output rays as color.

  /// // Initialize the color buffer to (0,0,0,0).
  /// float _color_buffer[4*c_width*c_height] = {0.0};   // Supposedly initializes all elements to 0.
  /// dray::Array<dray::Vec<float, 4>> color_buffer( (dray::Vec<float,4> *) _color_buffer, c_width*c_height);

  /// dray::ShadingContext<float> shading_ctx = mesh_field.get_shading_context(rays);

  /// {
  ///   // Hack: We are goint to colorize the hit ref pt.
  ///   const int *r_hit_idx_ptr = rays.m_hit_idx.get_host_ptr_const();
  ///   const dray::Vec<float,3> *r_hit_ref_pt_ptr = rays.m_hit_ref_pt.get_host_ptr_const();
  ///   dray::Vec<float,4> *img_ptr = color_buffer.get_host_ptr();
  ///   for (int ray_idx = 0; ray_idx < rays.size(); ray_idx++)
  ///   {
  ///     img_ptr[ray_idx][0] = /*(r_hit_idx_ptr[ray_idx] >= 0) ? 0.9 :*/ r_hit_ref_pt_ptr[ray_idx][0];
  ///     img_ptr[ray_idx][1] = /*(r_hit_idx_ptr[ray_idx] >= 0) ? 0.9 :*/ r_hit_ref_pt_ptr[ray_idx][1];
  ///     img_ptr[ray_idx][2] = /*(r_hit_idx_ptr[ray_idx] >= 0) ? 0.9 :*/ r_hit_ref_pt_ptr[ray_idx][2];
  ///     img_ptr[ray_idx][3] = (r_hit_idx_ptr[ray_idx] >= 0) ? 0.9 : 1.0;
  ///   }

  ///   dray::PNGEncoder png_encoder;
  ///   png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
  ///   png_encoder.save("identification.png");
  /// }
}

}

#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/mfem_mesh.hpp>
#include <dray/utils/timer.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/mfem_grid_function.hpp>
#include <dray/utils/mfem_utils.hpp>

#include <fstream>
#include <stdlib.h>

#include <mfem.hpp>
using namespace mfem;

int dim;
double freq = 1.0, kappa;

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}

void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

TEST(dray_mfem_test, dray_test_unit)
{
  //std::string file_name = std::string(DATA_DIR) + "beam-hex.mesh";
  std::string file_name = std::string(DATA_DIR) + "beam-hex-nurbs.mesh";
  std::cout<<"File name "<<file_name<<"\n";
  
  Mesh *mesh = new Mesh(file_name.c_str(), 1, 1);
  dim = mesh->Dimension();
  bool static_cond = false;
  int sdim = mesh->SpaceDimension();
  int order = 1;
  std::cout<<"Dim : "<<dim<<"\n"; //  Dims in referene space
  std::cout<<"Space Dim : "<<sdim<<"\n";

  //constexpr float max_els = 50000.f;
  constexpr float max_els = 500.f;
  // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(max_els/mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   mesh->ReorientTetMesh();

   // 4. Define a finite element space on the mesh. Here we use the Nedelec
   //    finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   VectorFunctionCoefficient f(sdim, f_exact);
   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   GridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   // 8. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   // 10. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //     solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 500, 1e-12, 0.0);
#else
   // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 11. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 12. Compute and print the L^2 norm of the error.
   cout << "\n|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E) << '\n' << endl;

   // 13. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      std::ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }
   
   //------- DRAY CODE --------
   dray::MFEMMesh h_mesh(mesh);
   h_mesh.print_self();

   const int psize = 1;
   const int mod = 1000000;
   dray::Array<dray::Vec3f> points;
   points.resize(psize);
   dray::Vec3f *points_ptr = points.get_host_ptr();

   // pick a bunch of random points inside the data bounds
   dray::AABB bounds = h_mesh.get_bounds();
   float x_length = bounds.m_x.length();
   float y_length = bounds.m_y.length();
   float z_length = bounds.m_z.length();
  
   for(int i = 0;  i < psize; ++i)
   {
     float x = ((rand() % mod) / float(mod)) * x_length - bounds.m_x.min();
     float y = ((rand() % mod) / float(mod)) * y_length - bounds.m_y.min();
     float z = ((rand() % mod) / float(mod)) * z_length - bounds.m_z.min();
     points_ptr[i][0] = x;
     points_ptr[i][1] = y;
     points_ptr[i][2] = z;
   }
   std::cout<<"locating\n";
   h_mesh.locate(points);
   
   // Get scalar field bounds.

   // Using mfem_utils grid_function_bounds().
   float field_lower, field_upper;
   dray::detail::grid_function_bounds(&x, 2, field_lower, field_upper);
   ///dray::detail::grid_function_bounds(mesh->GetNodes(), 2, field_lower, field_upper);
   std::cerr << "Refinement == 2, field values are within [" << field_lower << ", " << field_upper << "]" <<std::endl;
   /// dray::detail::grid_function_bounds(&x, 4, field_lower, field_upper);
   /// std::cerr << "Refinement == 4, field_lower == " << field_lower << std::endl;
   /// std::cerr << "Refinement == 4, field_upper == " << field_upper << std::endl;

   //--// Using MFEMGridFunction::get_bounds().
   /// float field_lower, field_upper;
   /// dray::MFEMGridFunction x_pos(&x);                     // Using the scalar field.
   /// //dray::MFEMGridFunction x_pos(mesh->GetNodes());      // Test using the mesh geometry grid function instead.
   /// x_pos.get_bounds(field_lower, field_upper);
   /// std::cout << "field_lower == " << field_lower << std::endl;
   /// std::cout << "field_upper == " << field_upper << std::endl;

   //----- end DRAY CODE ------

   // 15. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   DRAY_LOG_WRITE("mfem");
}

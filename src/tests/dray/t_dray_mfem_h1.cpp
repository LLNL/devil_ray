#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/mfem_mesh.hpp>
#include <dray/mfem_volume_integrator.hpp>
#include <dray/utils/timer.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/mfem_grid_function.hpp>

#include <fstream>
#include <stdlib.h>

#include <mfem.hpp>
using namespace mfem;

TEST(dray_mfem_h1_test, dray_test_unit)
{
  //std::string file_name = std::string(DATA_DIR) + "beam-hex.mesh";
  std::string file_name = std::string(DATA_DIR) + "beam-hex-nurbs.mesh";
  std::cout<<"File name "<<file_name<<"\n";
  
  Mesh *mesh = new Mesh(file_name.c_str(), 1, 1);
  int dim = mesh->Dimension();
  bool static_cond = false;
  int sdim = mesh->SpaceDimension();
  int order = 1;
  std::cout<<"Dim : "<<dim<<"\n"; //  Dims in referene space
  std::cout<<"Space Dim : "<<sdim<<"\n";

  constexpr float max_els = 50000.f;
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
   
   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (mesh->GetNodes())
   {
      fec = mesh->GetNodes()->OwnFEC();
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
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

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

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

   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // 11. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);
   
   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   //------- DRAY CODE --------
   //dray::MFEMMesh h_mesh(mesh);
   dray::MFEMMeshField h_mesh(mesh, &x);
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
   dray::Array<dray::int32> elt_ids;
   dray::Array<dray::Vec<float,3>> ref_pts;
   elt_ids.resize(psize);
   ref_pts.resize(psize);
   h_mesh.locate(points, elt_ids, ref_pts);

   // Get scalar field bounds.
   // Using MFEMGridFunction::get_bounds().
   float field_lower, field_upper;
   //dray::MFEMGridFunction x_pos(&x);                     // Using the scalar field.
   //dray::MFEMGridFunction x_pos(mesh->GetNodes());      // Test using the mesh geometry grid function instead.
   //x_pos.field_bounds(field_lower, field_upper);
   h_mesh.field_bounds(field_lower, field_upper);
   std::cout << "field values are within [" << field_lower << ", " << field_upper << "]" << std::endl;

   // Volume rendering.
   dray::Camera camera;
   camera.set_width(10);
   camera.set_height(10);
   camera.reset_to_bounds(h_mesh.get_bounds());
   dray::ray32 rays;
   camera.create_rays(rays);
   dray::MFEMVolumeIntegrator integrator(h_mesh);
   dray::Array<dray::Vec<dray::float32,4>> color_buffer = integrator.integrate(rays);

   //----- end DRAY CODE ------
   
   // 14. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   DRAY_LOG_WRITE("mfem");
}

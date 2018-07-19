#include "gtest/gtest.h"
#include "test_config.h"

#include <dray/mfem2dray.hpp>
#include <mfem.hpp>

#include <dray/camera.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>


// Helper function prototype.

// Returns pointer to new mesh. Parameter sol is not new.
// Caller is responsible to delete mesh_ptr.
void construct_example_data(const int num_el, mfem::Mesh *mesh_ptr, mfem::GridFunction &sol);

//
// TEST()
//
TEST(dray_test, dray_mfem_reader)
{
  mfem::Mesh *mfem_mesh_ptr;
  mfem::GridFunction mfem_sol;

  // Initialize mfem data.
  //construct_example_data(50000, mfem_mesh_ptr, mfem_sol);
  construct_example_data(500, mfem_mesh_ptr, mfem_sol);

  //DEBUG
  printf("before GetNodes().\n");
  mfem_mesh_ptr->GetNodes();
  printf("after GetNodes().\n");

  // --- DRAY code --- //

  dray::ElTransData<float,3> space_data = dray::import_mesh<float>(*mfem_mesh_ptr);

  space_data.m_ctrl_idx.summary();
  space_data.m_values.summary();

  //TODO dray::MeshField

  // --- end DRAY  --- //

  delete mfem_mesh_ptr;
}


// --- MFEM code --- //

void construct_example_data(const int in_max_els, mfem::Mesh *out_mesh_ptr, mfem::GridFunction &out_sol)
{
  using namespace mfem;

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

  const float max_els = in_max_els;
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


   // Output to arguments.
   out_mesh_ptr = mesh;
   out_sol = x;

}

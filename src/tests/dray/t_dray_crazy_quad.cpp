#include "gtest/gtest.h"
#include "test_config.h"

#include "t_utils.hpp"
#include <mfem.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/io/mfem_reader.hpp>

#include <fstream>
#include <stdlib.h>


// Helper function prototype.

// Returns pointer to new mesh and grid function.
// Caller is responsible to delete mesh_ptr and sol.
void construct_example_data(const int num_el,
                            mfem::Mesh *&mesh_ptr,
                            mfem::GridFunction * &sol,
                            int order = 2);


#if 0
// TEST()
TEST(dray_crazy_quad, dray_crazy_quad_convert)
{
  std::string file_name = std::string(DATA_DIR) + "crazy_hex/crazy_hex";
  std::string output_visit_dc = "crazy_hex_positive";
  /// std::string file_name = std::string(DATA_DIR) + "warbly_cube/warbly_cube";
  /// std::string output_visit_dc = "warbly_cube_positive";
  std::string output_path = conduit::utils::join_file_path(prepare_output_dir(), output_visit_dc);

  mfem::Mesh *mfem_mesh_ptr;
  mfem::GridFunction *mfem_sol_ptr;

  mfem::ConduitDataCollection dcol(file_name);
  dcol.SetProtocol("conduit_bin");
  dcol.Load();
  mfem_mesh_ptr = dcol.GetMesh();
  mfem_sol_ptr = dcol.GetField("bananas");

  if (mfem_mesh_ptr->NURBSext)
  {
     mfem_mesh_ptr->SetCurvature(20);
     /// mfem_mesh_ptr->SetCurvature(4);
  }


  // Convert to positive basis.

  bool is_mesh_gf_new;
  bool is_field_gf_new;

  mfem::GridFunction *mesh_nodes = mfem_mesh_ptr->GetNodes();
  if (mesh_nodes == nullptr)
  {
    std::cerr << "Could not get mesh nodes.\n";
    assert(false);
  }

  // Create a new FECollection and project mesh nodes, if the node grid function
  // does not already use a positive basis.
  // The positive FECollection can be accessed through pos_mesh_nodes.FESpace()->FEColl();
  mfem::GridFunction *pos_mesh_nodes_ptr = dray::project_to_pos_basis(mesh_nodes, is_mesh_gf_new);
  mfem::GridFunction & pos_mesh_nodes = (is_mesh_gf_new ? *pos_mesh_nodes_ptr : *mesh_nodes);

  // Use the new node grid function that lives on the positive FECollection.
  // We are responsible to make sure that the old nodes, which will be deleted,
  // are not the same as the new nodes.
  if (&pos_mesh_nodes != mfem_mesh_ptr->GetNodes())
    mfem_mesh_ptr->NewNodes(pos_mesh_nodes, true);

  // Get a grid function that lives on a postive FECollection.
  // If the original grid function did not, then create a new FESpace over our
  // new positive FECollection, and use that to create a new grid function,
  // onto which we can project the old grid function.
  //
  // This more or less duplicates the logic of dray::project_to_pos_basis(), except it
  // re-uses the same FECollection for both the mesh nodes and the field.
  mfem::GridFunction *pos_field_ptr;
  mfem::FiniteElementSpace *pos_field_fe_space;
  if (is_mesh_gf_new)
  {
    pos_field_fe_space = new mfem::FiniteElementSpace(mfem_mesh_ptr,
        mfem_mesh_ptr->GetNodes()->FESpace()->FEColl(),
        mfem_sol_ptr->FESpace()->GetVDim());
    pos_field_ptr = new mfem::GridFunction(pos_field_fe_space); 

    if (pos_field_ptr == nullptr)
    {
      std::cerr << "Could not create new GridFunction with positive FESpace.\n";
      assert(false);
    }

    pos_field_ptr->ProjectGridFunction(*mfem_sol_ptr);
  }
  else
    pos_field_ptr = mfem_sol_ptr;

  // Save to Visit data collection.

  mfem::VisItDataCollection visit_dc(output_visit_dc, mfem_mesh_ptr);
  visit_dc.SetPrefixPath(output_path);
  visit_dc.RegisterField("positive_bananas",  pos_field_ptr);
  visit_dc.SetCycle(0);
  visit_dc.SetTime(0.0);
  visit_dc.Save();

  if (is_mesh_gf_new)
  {
    delete pos_field_ptr;
    delete pos_field_fe_space;
  }
}
#endif



TEST(dray_crazy_quad, dray_crazy_quad_save_mesh)
{
  constexpr int order = 20;

  std::string output_visit_dc = "CrazyQuadData";
  std::string output_path = conduit::utils::join_file_path(prepare_output_dir(), output_visit_dc);

  mfem::Mesh * new_mesh;
  mfem::GridFunction * new_sol;
  construct_example_data(1, new_mesh, new_sol, order);

  mfem::VisItDataCollection visit_dc(output_visit_dc, new_mesh);
  visit_dc.SetPrefixPath(output_path);
  visit_dc.RegisterField("bananas",  new_sol);
  visit_dc.SetCycle(0);
  visit_dc.SetTime(0.0);
  visit_dc.Save();

  delete new_mesh;
  delete new_sol;
}



void construct_example_data(const int in_max_els,
                            mfem::Mesh *&out_mesh_ptr,
                            mfem::GridFunction * &out_sol_ptr,
                            int order)
{
  using namespace mfem;

  std::string file_name = std::string(DATA_DIR) + "quad-spiral-q20.mesh";
  /// std::string file_name = std::string(DATA_DIR) + "beam-hex.mesh";
  //std::string file_name = std::string(DATA_DIR) + "beam-hex-nurbs.mesh";
  //std::string file_name = std::string(DATA_DIR) + "spiral_hex_p20.mesh";
  std::cout<<"File name "<<file_name<<"\n";

  Mesh *mesh = new Mesh(file_name.c_str(), 1, 1);
  int dim = mesh->Dimension();
  bool static_cond = false;
  int sdim = mesh->SpaceDimension();
  std::cout<<"Dim : "<<dim<<"\n"; //  Dims in referene space
  std::cout<<"Space Dim : "<<sdim<<"\n";

  const float max_els = in_max_els;
  /// // 3. Refine the mesh to increase the resolution. In this example we do
  /// //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  /// //    largest number that gives a final mesh with no more than 50,000
  /// //    elements.
  /// {
  ///    int ref_levels =
  ///       (int)floor(log(max_els/mesh->GetNE())/log(2.)/dim);
  ///    for (int l = 0; l < ref_levels; l++)
  ///    {
  ///       mesh->UniformRefinement();
  ///    }
  /// }

   /// mesh->ReorientTetMesh();

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
   GridFunction *_x = new GridFunction(fespace);
   GridFunction &x = *_x;
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
   out_sol_ptr = _x;

   printf("In construct_example(): fespace == %x\n", fespace);

   // TODO didn't there used to be some "delete" statements?
}


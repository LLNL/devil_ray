#include "gtest/gtest.h"
#include "test_config.h"

#include "t_utils.hpp"
#include <mfem.hpp>
#include <mfem/fem/conduitdatacollection.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/filters/volume_integrator.hpp>

#include <dray/camera.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/utils/ray_utils.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>


// Helper function prototype.

// Returns pointer to new mesh and grid function.
// Caller is responsible to delete mesh_ptr and sol.
void construct_example_data(const int num_el,
                            mfem::Mesh *&mesh_ptr,
                            mfem::GridFunction * &sol,
                            int order = 2,
                            std::string fname = "beam-hex.mesh");

TEST(dray_convert_mesh, dray_convert)
{
  mfem::Mesh *mesh_ptr;
  mfem::GridFunction *field_ptr;
  construct_example_data(4, mesh_ptr, field_ptr);

  mfem::ConduitDataCollection col("test_mesh");
  col.SetMesh(mesh_ptr);
  col.RegisterField("test_field", field_ptr);
  col.SetProtocol("conduit_json");
  col.Save();
}

TEST(dray_volume_render, dray_volume_render_simple)
{
  std::string file_name = std::string(DATA_DIR) + "impeller/impeller";
  int cycle = 0;
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "impeller_vr");
  remove_test_image(output_file);

  using MeshElemT = dray::MeshElem<float, 3u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;
  dray::DataSet<float, MeshElemT> dataset = dray::MFEMReader::load32(file_name, cycle);

  dray::ColorTable color_table("Spectral");
  color_table.add_alpha(0.f,  0.01f);
  color_table.add_alpha(0.1f, 0.09f);
  color_table.add_alpha(0.2f, 0.01f);
  color_table.add_alpha(0.3f, 0.09f);
  color_table.add_alpha(0.4f, 0.01f);
  color_table.add_alpha(0.5f, 0.01f);
  color_table.add_alpha(0.6f, 0.01f);
  color_table.add_alpha(0.7f, 0.09f);
  color_table.add_alpha(0.8f, 0.01f);
  color_table.add_alpha(0.9f, 0.01f);
  color_table.add_alpha(1.0f, 0.0f);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;
  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.reset_to_bounds(dataset.get_mesh().get_bounds());
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  dray::VolumeIntegrator integrator;
  integrator.set_field("bananas");
  integrator.set_color_table(color_table);
  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  color_buffer = integrator.execute(rays, dataset);

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      camera.get_width(),
                      camera.get_height() );
  png_encoder.save(output_file + ".png");
  EXPECT_TRUE(check_test_image(output_file));
}

TEST(dray_volume_render, dray_volume_render_triple)
{
  std::string file_name = std::string(DATA_DIR) + "tripple_point/field_dump.cycle";
  int cycle = 0;
  cycle = 6700;
  std::string output_path = prepare_output_dir();
  std::string output_file = conduit::utils::join_file_path(output_path, "triple_vr");
  remove_test_image(output_file);

  using MeshElemT = dray::MeshElem<float, 3u, dray::ElemType::Quad, dray::Order::General>;
  using FieldElemT = dray::FieldOn<MeshElemT, 1u>;
  dray::DataSet<float, MeshElemT> dataset = dray::MFEMReader::load32(file_name, cycle);

  dray::ColorTable color_table("Spectral");
  color_table.add_alpha(0.f,  0.01f);
  color_table.add_alpha(0.1f, 0.09f);
  color_table.add_alpha(0.2f, 0.01f);
  color_table.add_alpha(0.3f, 0.09f);
  color_table.add_alpha(0.4f, 0.01f);
  color_table.add_alpha(0.5f, 0.01f);
  color_table.add_alpha(0.6f, 0.01f);
  color_table.add_alpha(0.7f, 0.09f);
  color_table.add_alpha(0.8f, 0.09f);
  color_table.add_alpha(0.9f, 0.01f);
  color_table.add_alpha(1.0f, 0.1f);

  // Camera
  const int c_width = 1024;
  const int c_height = 1024;
  dray::Camera camera;
  camera.set_width(c_width);
  camera.set_height(c_height);
  camera.reset_to_bounds(dataset.get_mesh().get_bounds());
  dray::Array<dray::ray32> rays;
  camera.create_rays(rays);

  dray::VolumeIntegrator integrator;
  integrator.set_field("density");
  integrator.set_color_table(color_table);
  dray::Array<dray::Vec<dray::float32,4>> color_buffer;
  color_buffer = integrator.execute(rays, dataset);

  dray::PNGEncoder png_encoder;
  png_encoder.encode( (float *) color_buffer.get_host_ptr(),
                      camera.get_width(),
                      camera.get_height() );
  png_encoder.save(output_file + ".png");
  EXPECT_TRUE(check_test_image(output_file));
}


// --- MFEM code --- //

void construct_example_data(const int in_max_els,
                            mfem::Mesh *&out_mesh_ptr,
                            mfem::GridFunction * &out_sol_ptr,
                            int order,
                            std::string fname)
{
  using namespace mfem;

  std::string file_name = std::string(DATA_DIR) + fname;
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

#include "gtest/gtest.h"
#include "test_config.h"

#include "t_utils.hpp"
#include <mfem.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/io/mfem_reader.hpp>

#include <fstream>
#include <stdlib.h>


TEST(dray_crazy_hex, dray_crazy_hex_convert)
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



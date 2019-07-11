#include "gtest/gtest.h"
#include "test_config.h"

#include <mfem.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/io/mfem_reader.hpp>

#include <fstream>
#include <stdlib.h>


TEST(dray_crazy_hex, dray_crazy_hex_convert)
{
  std::string file_name = std::string(DATA_DIR) + "crazy_hex/crazy_hex";
  /// std::string output_path = prepare_output_dir();
  /// std::string output_file = conduit::utils::join_file_path(output_path, "crazy_hex_positive");

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

  mfem::GridFunction *pos_mesh_nodes_ptr = dray::project_to_pos_basis(mesh_nodes, is_mesh_gf_new);
  mfem::GridFunction & pos_mesh_nodes = (is_mesh_gf_new ? *pos_mesh_nodes_ptr : *mesh_nodes);

  mfem_mesh_ptr->NewNodes(pos_mesh_nodes);

  mfem::GridFunction *pos_field_ptr = dray::project_to_pos_basis(mfem_sol_ptr, is_field_gf_new);
  mfem::GridFunction & pos_field = (is_field_gf_new ? *pos_field_ptr : *mfem_sol_ptr);


  // Save to Visit data collection.

  mfem::VisItDataCollection visit_dc("crazy_hex_positive", mfem_mesh_ptr);
  visit_dc.RegisterField("positive_bananas",  &pos_field);
  visit_dc.SetCycle(0);
  visit_dc.SetTime(0.0);
  visit_dc.Save();

  if (is_mesh_gf_new)
    delete pos_mesh_nodes_ptr;

  if (is_field_gf_new)
    delete pos_field_ptr;
}



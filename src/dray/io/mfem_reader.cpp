// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/error.hpp>
#include <dray/derived_topology.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/mfem_reader.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/utils/data_logger.hpp>

#include <mfem/fem/conduitdatacollection.hpp>

namespace dray
{

namespace detail
{

mfem::DataCollection *load_collection (const std::string root_file, const int32 cycle)
{
  // start with visit
  mfem::VisItDataCollection *vcol = new mfem::VisItDataCollection (root_file);
  try
  {
    vcol->Load (cycle);
    // apparently failing to open is just a warning...
    if (vcol->GetMesh () == nullptr)
    {
      throw DRayError ("Failed");
    }
    DRAY_INFO ("Load succeeded 'visit data collection'");
    return vcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'visit data collection'");
  }
  delete vcol;

  // now try conduit
  mfem::ConduitDataCollection *dcol = new mfem::ConduitDataCollection (root_file);
  try
  {
    dcol->SetProtocol ("conduit_bin");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'conduit_bin'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'conduit_bin'");
  }

  try
  {
    dcol->SetProtocol ("conduit_json");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'conduit_json'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'conduit_json'");
  }

  try
  {
    dcol->SetProtocol ("json");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'json'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'json'");
  }

  try
  {
    dcol->SetProtocol ("hdf5");
    dcol->Load (cycle);
    DRAY_INFO ("Load succeeded 'hdf5'");
    return dcol;
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'hdf5'");
  }


  delete dcol;

  return nullptr;
}

DataSet load(const std::string &root_file, const int32 cycle)
{
  using MeshElemT = MeshElem<3u, Quad, General>;
  using FieldElemT = FieldOn<MeshElemT, 1u>;

  mfem::DataCollection *dcol = load_collection (root_file, cycle);
  if (dcol == nullptr)
  {
    throw DRayError ("Failed to open file '" + root_file + "'");
  }

  mfem::Mesh *mfem_mesh_ptr;

  mfem_mesh_ptr = dcol->GetMesh ();

  if (mfem_mesh_ptr->NURBSext)
  {
    mfem_mesh_ptr->SetCurvature (2);
  }

  mfem_mesh_ptr->GetNodes ();
  int space_p;
  GridFunction<3> space_data = dray::import_mesh (*mfem_mesh_ptr, space_p);
  Mesh<MeshElemT> mesh (space_data, space_p);

  std::shared_ptr<HexTopology> topo = std::make_shared<HexTopology>(mesh);
  DataSet dataset(topo);

  auto field_map = dcol->GetFieldMap ();
  for (auto it = field_map.begin (); it != field_map.end (); ++it)
  {
    const std::string field_name = it->first;
    mfem::GridFunction *grid_ptr = dcol->GetField (field_name);
    const int components = grid_ptr->VectorDim ();

    const mfem::FiniteElementSpace *fespace = grid_ptr->FESpace ();
    const int32 P = fespace->GetOrder (0);
    if (P == 0)
    {
      DRAY_INFO ("Field has unsupported order " << P);
      continue;
    }
    if (components == 1)
    {
      int field_p;
      GridFunction<1> field_data = dray::import_grid_function<1> (*grid_ptr, field_p);
      Field<FieldElemT> field (field_data, field_p);

      std::shared_ptr<Field<FieldElemT>> ffield
        = std::make_shared<Field<FieldElemT>>(field);
      dataset.add_field(ffield);
    }
    else if (components == 3)
    {
      Field<FieldElemT> field_x =
        import_vector_field_component<MeshElemT> (*grid_ptr, 0);
      field_x.name(field_name + "_x");

      Field<FieldElemT> field_y =
        import_vector_field_component<MeshElemT> (*grid_ptr, 1);
      field_y.name(field_name + "_y");

      Field<FieldElemT> field_z =
        import_vector_field_component<MeshElemT> (*grid_ptr, 2);
      field_z.name(field_name + "_z");

      std::shared_ptr<Field<FieldElemT>> ffield_x
        = std::make_shared<Field<FieldElemT>>(field_x);
      dataset.add_field(ffield_x);

      std::shared_ptr<Field<FieldElemT>> ffield_y
        = std::make_shared<Field<FieldElemT>>(field_y);
      dataset.add_field(ffield_y);

      std::shared_ptr<Field<FieldElemT>> ffield_z
        = std::make_shared<Field<FieldElemT>>(field_z);
      dataset.add_field(ffield_z);
    }
    else
    {
      std::cout << "Import field: number of components = " << components << " not supported \n";
    }
  }

  delete dcol;
  return dataset;
}


} // namespace detail

DataSet
MFEMReader::load (const std::string &root_file, const int32 cycle)
{
  try
  {
    return detail::load (root_file, cycle);
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'mfem data collection'");
  }
  try
  {
    return BlueprintReader::load (root_file, cycle);
  }
  catch (...)
  {
    DRAY_INFO ("Load failed 'blueprint reader'");
  }

  throw DRayError ("Failed to open file '" + root_file + "'");
}
// TODO triangle, 2d, etc.

} // namespace dray

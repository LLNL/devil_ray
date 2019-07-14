#include <dray/io/mfem_reader.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/error.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/utils/data_logger.hpp>

#include <mfem/fem/conduitdatacollection.hpp>

namespace dray
{

namespace detail
{

mfem::DataCollection *
load_collection(const std::string root_file, const int32 cycle)
{
  // start with visit
  mfem::VisItDataCollection *vcol = new mfem::VisItDataCollection(root_file);
  try
  {
    vcol->Load(cycle);
    // apparently failing to open is just a warning...
    if(vcol->GetMesh() == nullptr)
    {
      throw DRayError("Failed");
    }
    DRAY_INFO("Load succeeded 'visit data collection'");
    return vcol;
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'visit data collection'");
  }
  delete vcol;

  // now try conduit
  mfem::ConduitDataCollection *dcol = new mfem::ConduitDataCollection(root_file);
  try
  {
    dcol->SetProtocol("conduit_bin");
    dcol->Load(cycle);
    DRAY_INFO("Load succeeded 'conduit_bin'");
    return dcol;
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'conduit_bin'");
  }

  try
  {
    dcol->SetProtocol("conduit_json");
    dcol->Load(cycle);
    DRAY_INFO("Load succeeded 'conduit_json'");
    return dcol;
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'conduit_json'");
  }

  try
  {
    dcol->SetProtocol("json");
    dcol->Load(cycle);
    DRAY_INFO("Load succeeded 'json'");
    return dcol;
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'json'");
  }

  try
  {
    dcol->SetProtocol("hdf5");
    dcol->Load(cycle);
    DRAY_INFO("Load succeeded 'hdf5'");
    return dcol;
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'hdf5'");
  }


  delete dcol;

  return nullptr;
}

template<typename T>
DataSet<T> load(const std::string &root_file, const int32 cycle)
{
  mfem::DataCollection *dcol = load_collection(root_file, cycle);
  if(dcol == nullptr)
  {
    throw DRayError("Failed to open file '" + root_file + "'");
  }

  mfem::Mesh *mfem_mesh_ptr;

  mfem_mesh_ptr = dcol->GetMesh();

  if (mfem_mesh_ptr->NURBSext)
  {
     mfem_mesh_ptr->SetCurvature(2);
  }

  mfem_mesh_ptr->GetNodes();
  int space_p;
  dray::ElTransData<T,3> space_data = dray::import_mesh<T>(*mfem_mesh_ptr, space_p);
  dray::Mesh<T> mesh(space_data, space_p);

  DataSet<T> dataset(mesh);

  auto field_map = dcol->GetFieldMap();
  for(auto it = field_map.begin(); it != field_map.end(); ++it)
  {
    const std::string field_name = it->first;
    mfem::GridFunction *grid_ptr = dcol->GetField(field_name);
    const int components = grid_ptr->VectorDim();
    if(components == 1)
    {
      int field_p;
      ElTransData<T,1> field_data = dray::import_grid_function<T,1>(*grid_ptr, field_p);
      Field<T> field(field_data, field_p);
      dataset.add_field(field, field_name);
    }
    else if(components == 3)
    {
      dray::Field<T> field_x = dray::import_vector_field_component<T>(*grid_ptr, 0);
      dray::Field<T> field_y = dray::import_vector_field_component<T>(*grid_ptr, 1);
      dray::Field<T> field_z = dray::import_vector_field_component<T>(*grid_ptr, 2);

      dataset.add_field(field_x, field_name + "_x");
      dataset.add_field(field_y, field_name + "_y");
      dataset.add_field(field_z, field_name + "_z");
    }
    else
    {
      std::cout<<"Import field: number of components = "<<components
               <<" not supported \n";
    }
  }

  delete dcol;
  return dataset;
}


} // namespace detail

DataSet<float32>
MFEMReader::load32(const std::string &root_file, const int32 cycle)
{
  try
  {
    return detail::load<float32>(root_file, cycle);
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'mfem data collection'");
  }
  try
  {
    return BlueprintReader::load32(root_file, cycle);
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'blueprint reader'");
  }

  throw DRayError("Failed to open file '" + root_file + "'");
}

DataSet<float64>
MFEMReader::load64(const std::string &root_file, const int32 cycle)
{
  try
  {
    return detail::load<float64>(root_file, cycle);
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'mfem data collection'");
  }
  try
  {
    return BlueprintReader::load64(root_file, cycle);
  }
  catch(...)
  {
    DRAY_INFO("Load failed 'blueprint reader'");
  }

  throw DRayError("Failed to open file '" + root_file + "'");
}

} //namespace dray

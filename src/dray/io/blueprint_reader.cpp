#include <dray/io/blueprint_reader.hpp>
#include <dray/error.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/utils/data_logger.hpp>

#include <mfem/fem/conduitdatacollection.hpp>
// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <fstream>

using namespace conduit;

namespace dray
{
namespace detail
{
class BlueprintTreePathGenerator
{
public:
BlueprintTreePathGenerator(const std::string &file_pattern,
                           const std::string &tree_pattern,
                           int num_files,
                           int num_trees,
                           const std::string &protocol,
                           const Node &mesh_index)
: m_file_pattern(file_pattern),
  m_tree_pattern(tree_pattern),
  m_num_files(num_files),
  m_num_trees(num_trees),
  m_protocol(protocol),
  m_mesh_index(mesh_index)
{

}

//-------------------------------------------------------------------//
~BlueprintTreePathGenerator()
{

}

//-------------------------------------------------------------------//
std::string Expand(const std::string pattern,
                   int idx) const
{
    //
    // Note: This currently only handles format strings:
    // "%05d" "%06d" "%07d"
    //

    std::size_t pattern_idx = pattern.find("%05d");

    if(pattern_idx != std::string::npos)
    {
        char buff[16];
        snprintf(buff,16,"%05d",idx);
        std::string res = pattern;
        res.replace(pattern_idx,4,std::string(buff));
        return res;
    }

    pattern_idx = pattern.find("%06d");

    if(pattern_idx != std::string::npos)
    {
        char buff[16];
        snprintf(buff,16,"%06d",idx);
        std::string res = pattern;
        res.replace(pattern_idx,4,std::string(buff));
        return res;
    }

    pattern_idx = pattern.find("%07d");

    if(pattern_idx != std::string::npos)
    {
        char buff[16];
        snprintf(buff,16,"%07d",idx);
        std::string res = pattern;
        res.replace(pattern_idx,4,std::string(buff));
        return res;
    }
    return pattern;
}


//-------------------------------------------------------------------//
std::string GenerateFilePath(int tree_id) const
{
    // for now, we only support 1 tree per file.
    int file_id = tree_id;
    return Expand(m_file_pattern,file_id);
}

//-------------------------------------------------------------------//
std::string GenerateTreePath(int tree_id) const
{
    // the tree path should always end in a /
    std::string res = Expand(m_tree_pattern,tree_id);
    if( (res.size() > 0) && (res[res.size()-1] != '/') )
    {
        res += "/";
    }
    return res;
}

private:
  std::string m_file_pattern;
  std::string m_tree_pattern;
  int m_num_files;
  int m_num_trees;
  std::string m_protocol;
  Node m_mesh_index;

};

//-----------------------------------------------------------------------------
void relay_blueprint_mesh_read(const Node &options,
                               Node &data)
{
    std::string root_fname = options["root_file"].as_string();

    // read the root file, it can be either json or hdf5

    // assume hdf5, but check for json file
    std::string root_protocol = "hdf5";
    char buff[5] = {0,0,0,0,0};

    // heuristic, if json, we expect to see "{" in the first 5 chars of the file.
    std::ifstream ifs;
    ifs.open(root_fname.c_str());
    if(!ifs.is_open())
    {
       throw DRayError("failed to open relay root file: " + root_fname);
    }
    ifs.read((char *)buff,5);
    ifs.close();

    std::string test_str(buff);

    if(test_str.find("{") != std::string::npos)
    {
       root_protocol = "json";
    }

    Node root_node;
    relay::io::load(root_fname, root_protocol, root_node);


    if(!root_node.has_child("file_pattern"))
    {
        throw DRayError("Root file missing 'file_pattern'");
    }

    if(!root_node.has_child("blueprint_index"))
    {
        throw DRayError("Root file missing 'blueprint_index'");
    }

    NodeConstIterator itr =root_node["blueprint_index"].children();
    Node verify_info;
    // TODO, for now lets verify the first mesh index

    const Node &mesh_index = itr.next();

    if( !blueprint::mesh::index::verify(mesh_index,
                                        verify_info[itr.name()]))
    {
      std::cout<<"Mesh Blueprint index verify failed" << std::endl
               << verify_info.to_json()<<"\n";
    }

    std::string data_protocol = "hdf5";

    if(root_node.has_child("protocol"))
    {
        data_protocol = root_node["protocol/name"].as_string();
    }

    // read the first mesh (all domains ...)

    int num_domains = root_node["number_of_trees"].to_int();
    if(num_domains != 1)
    {
      throw DRayError("only supports single domain");
    }

    BlueprintTreePathGenerator gen(root_node["file_pattern"].as_string(),
                                   root_node["tree_pattern"].as_string(),
                                   root_node["number_of_files"].to_int(),
                                   num_domains,
                                   data_protocol,
                                   mesh_index);

    std::ostringstream oss;

    char domain_fmt_buff[64];
    int domain_id = 0;
    snprintf(domain_fmt_buff, sizeof(domain_fmt_buff), "%06d",domain_id);
    oss.str("");
    oss << "domain_" << std::string(domain_fmt_buff);

    std::string current, next;
    utils::rsplit_file_path(root_fname, current, next);
    std::string domain_file = utils::join_path(next,gen.GenerateFilePath(domain_id));
    relay::io::load(domain_file,
                    data_protocol,
                    data);
}

std::string append_cycle(const std::string &base, const int cycle)
{
  std::ostringstream oss;

  char fmt_buff[64];
  snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);
  oss.str("");
  oss << base << "_" << std::string(fmt_buff);
  return oss.str();
}
template<typename T>
DataSet<T> bp2dray(conduit::Node &n_dataset)
{
  mfem::Mesh *mfem_mesh_ptr = mfem::ConduitDataCollection::BlueprintMeshToMesh(n_dataset);

  mfem_mesh_ptr->GetNodes();
  int space_p;
  dray::ElTransData<T,3> space_data = dray::import_mesh<T>(*mfem_mesh_ptr, space_p);

  dray::Mesh<T> mesh(space_data, space_p);
  DataSet<T> dataset(mesh);

  NodeConstIterator itr = n_dataset["fields"].children();

  std::string nodes_gf_name = "";

  const Node &n_topo = n_dataset["topologies/main"];
  if (n_topo.has_child("grid_function"))
  {
     nodes_gf_name = n_topo["grid_function"].as_string();
  }

  while (itr.has_next())
  {
     const Node &n_field = itr.next();
     std::string field_name = itr.name();

     // skip mesh nodes gf since they are already processed
     // skip attribute fields, they aren't grid functions
     if ( field_name != nodes_gf_name &&
          field_name.find("_attribute") == std::string::npos
        )
     {
       mfem::GridFunction *grid_ptr =
         mfem::ConduitDataCollection::BlueprintFieldToGridFunction(mfem_mesh_ptr, n_field);

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
  }
  return dataset;
}

template<typename T>
DataSet<T> load_bp(const std::string &root_file, const int cycle)
{
  Node options, data;
  options["root_file"] = append_cycle(root_file, cycle);

  std::cout<<"Trying blueprint file "<<append_cycle(root_file, cycle)<<"\n";;
  detail::relay_blueprint_mesh_read(options,data);
  data.schema().print();

  return bp2dray<T>(data);
}

} // namespace detail

DataSet<float32>
BlueprintReader::load32(const std::string &root_file, const int cycle)
{
  return detail::load_bp<float32>(root_file, cycle);
}

DataSet<float64>
BlueprintReader::load64(const std::string &root_file, const int cycle)
{
  return detail::load_bp<float64>(root_file, cycle);
}

DataSet<float32>
BlueprintReader::blueprint_to_dray32(conduit::Node &n_dataset)
{
  return detail::bp2dray<float32>(n_dataset);
}

DataSet<float64>
BlueprintReader::blueprint_to_dray64(conduit::Node &n_dataset)
{
  return detail::bp2dray<float64>(n_dataset);
}

} //namespace dray

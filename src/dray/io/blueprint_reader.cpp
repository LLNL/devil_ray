// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/error.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/derived_topology.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/utils/data_logger.hpp>

#include <mfem/fem/conduitdatacollection.hpp>
// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <fstream>

using namespace conduit;

namespace dray
{
namespace detail
{

std::string append_cycle (const std::string &base, const int cycle)
{
  std::ostringstream oss;

  char fmt_buff[64];
  snprintf (fmt_buff, sizeof (fmt_buff), "%06d", cycle);
  oss.str ("");
  oss << base << "_" << std::string (fmt_buff);
  return oss.str ();
}


class BlueprintTreePathGenerator
{
  public:
  BlueprintTreePathGenerator (const std::string &file_pattern,
                              const std::string &tree_pattern,
                              int num_files,
                              int num_trees,
                              const std::string &protocol,
                              const Node &mesh_index)
  : m_file_pattern (file_pattern), m_tree_pattern (tree_pattern),
    m_num_files (num_files), m_num_trees (num_trees), m_protocol (protocol),
    m_mesh_index (mesh_index)
  {
  }

  //-------------------------------------------------------------------//
  ~BlueprintTreePathGenerator ()
  {
  }

  //-------------------------------------------------------------------//
  std::string Expand (const std::string pattern, int idx) const
  {
    //
    // Note: This currently only handles format strings:
    // "%05d" "%06d" "%07d"
    //

    std::size_t pattern_idx = pattern.find ("%05d");

    if (pattern_idx != std::string::npos)
    {
      char buff[16];
      snprintf (buff, 16, "%05d", idx);
      std::string res = pattern;
      res.replace (pattern_idx, 4, std::string (buff));
      return res;
    }

    pattern_idx = pattern.find ("%06d");

    if (pattern_idx != std::string::npos)
    {
      char buff[16];
      snprintf (buff, 16, "%06d", idx);
      std::string res = pattern;
      res.replace (pattern_idx, 4, std::string (buff));
      return res;
    }

    pattern_idx = pattern.find ("%07d");

    if (pattern_idx != std::string::npos)
    {
      char buff[16];
      snprintf (buff, 16, "%07d", idx);
      std::string res = pattern;
      res.replace (pattern_idx, 4, std::string (buff));
      return res;
    }
    return pattern;
  }


  //-------------------------------------------------------------------//
  std::string GenerateFilePath (int tree_id) const
  {
    // for now, we only support 1 tree per file.
    int file_id = tree_id;
    return Expand (m_file_pattern, file_id);
  }

  //-------------------------------------------------------------------//
  std::string GenerateTreePath (int tree_id) const
  {
    // the tree path should always end in a /
    std::string res = Expand (m_tree_pattern, tree_id);
    if ((res.size () > 0) && (res[res.size () - 1] != '/'))
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

void relay_blueprint_mesh_read (const Node &options, Node &data)
{
  std::string full_root_fname = options["root_file"].as_string ();

  // read the root file, it can be either json or hdf5

  // assume hdf5, but check for json file
  std::string root_protocol = "hdf5";
  char buff[6] = { 0, 0, 0, 0, 0, 0};

  // heuristic, if json, we expect to see "{" in the first 5 chars of the file.
  std::ifstream ifs;
  ifs.open (full_root_fname.c_str ());
  if (!ifs.is_open ())
  {
    DRAY_ERROR ("failed to open relay root file: " + full_root_fname);
  }
  ifs.read ((char *)buff, 5);
  ifs.close ();

  std::string test_str (buff);

  if (test_str.find ("{") != std::string::npos)
  {
    root_protocol = "json";
  }

  Node root_node;
  relay::io::load (full_root_fname, root_protocol, root_node);

  if (!root_node.has_child ("file_pattern"))
  {
    DRAY_ERROR ("Root file missing 'file_pattern'");
  }

  if (!root_node.has_child ("blueprint_index"))
  {
    DRAY_ERROR ("Root file missing 'blueprint_index'");
  }

  NodeConstIterator itr = root_node["blueprint_index"].children ();
  Node verify_info;
  // TODO, for now lets verify the first mesh index

  const Node &mesh_index = itr.next ();

  if (!blueprint::mesh::index::verify (mesh_index, verify_info[itr.name ()]))
  {
    std::cout << "Mesh Blueprint index verify failed" << std::endl
              << verify_info.to_json () << "\n";
  }

  std::string data_protocol = "hdf5";

  if (root_node.has_child ("protocol"))
  {
    data_protocol = root_node["protocol/name"].as_string ();
  }

  // read the first mesh (all domains ...)

  int num_domains = root_node["number_of_trees"].to_int ();
  if (num_domains != 1)
  {
    DRAY_ERROR ("only supports single domain");
  }

  BlueprintTreePathGenerator gen (root_node["file_pattern"].as_string (),
                                  root_node["tree_pattern"].as_string (),
                                  root_node["number_of_files"].to_int (),
                                  num_domains, data_protocol, mesh_index);

  std::ostringstream oss;

  char domain_fmt_buff[64];
  int domain_id = 0;
  snprintf (domain_fmt_buff, sizeof (domain_fmt_buff), "%06d", domain_id);
  oss.str ("");
  oss << "domain_" << std::string (domain_fmt_buff);

  std::string current, next;
  utils::rsplit_file_path (full_root_fname, current, next);
  std::string domain_file = utils::join_path (next, gen.GenerateFilePath (domain_id));
  relay::io::load (domain_file, data_protocol, data);
}

//-----------------------------------------------------------------------------

template <typename T>
DataSet bp2dray (const conduit::Node &n_dataset, const ImportOrderPolicy &import_order_policy)
{
  using MeshElemT = MeshElem<3, Tensor, General>;
  using FieldElemT = FieldElem<3, 1, Tensor, General>;

  mfem::Mesh *mfem_mesh_ptr = mfem::ConduitDataCollection::BlueprintMeshToMesh (n_dataset);
  mfem::Geometry::Type geom_type = mfem_mesh_ptr->GetElementBaseGeometry(0);

  mfem_mesh_ptr->GetNodes ();

  DataSet dataset = import_mesh(*mfem_mesh_ptr, import_order_policy);

  NodeConstIterator itr = n_dataset["fields"].children ();

  std::string nodes_gf_name = "";
  std::string topo_name = "main";

  if (n_dataset["topologies"].number_of_children () == 0)
  {
    // this should not happen if verify is called before
    DRAY_ERROR ("Blueprint dataset has no topologies");
  }
  else
  {
    std::vector<std::string> names = n_dataset["topologies"].child_names ();
    topo_name = names[0];
  }

  const Node &n_topo = n_dataset["topologies/" + topo_name];
  if (n_topo.has_child ("grid_function"))
  {
    nodes_gf_name = n_topo["grid_function"].as_string ();
  }

  while (itr.has_next ())
  {
    const Node &n_field = itr.next ();
    std::string field_name = itr.name ();

    // skip mesh nodes gf since they are already processed
    // skip attribute fields, they aren't grid functions
    if (field_name != nodes_gf_name && field_name.find ("_attribute") == std::string::npos)
    {
      mfem::GridFunction *grid_ptr =
      mfem::ConduitDataCollection::BlueprintFieldToGridFunction (mfem_mesh_ptr, n_field);
      const mfem::FiniteElementSpace *fespace = grid_ptr->FESpace ();
      const int32 P = fespace->GetOrder (0);
      if (P == 0)
      {
        DRAY_WARN("Field has unsupported order " << P);
        continue;
      }
      const int components = grid_ptr->VectorDim ();
      if (components == 1)
      {
        DRAY_INFO("Importing field "<<field_name);

        int field_p;
        try
        {
          import_field(dataset, import_order_policy, *grid_ptr, geom_type, field_name);
        }
        catch(const DRayError &e)
        {
          DRAY_WARN("field import '"<<field_name<<"' failed with error '"
                    <<e.what()<<"'");
        }
      }
      else if (components == 3)
      {
        try
        {
          import_field(dataset, import_order_policy, *grid_ptr, geom_type, field_name + "_x", 0);
          import_field(dataset, import_order_policy, *grid_ptr, geom_type, field_name + "_y", 1);
          import_field(dataset, import_order_policy, *grid_ptr, geom_type, field_name + "_z", 2);
        }
        catch(const DRayError &e)
        {
          DRAY_WARN("vector field import '"<<field_name<<"' failed with error '"
                    <<e.what()<<"'");
        }
      }
      else
      {
        DRAY_INFO ("Import field: number of components = " << components << " not supported");
      }
      delete grid_ptr;
      DRAY_INFO ("Imported field name " << field_name);
    }
  }
  delete mfem_mesh_ptr;
  return dataset;
}

DataSet load_bp (const std::string &root_file, const ImportOrderPolicy &import_order_policy)
{
  Node options, data;
  options["root_file"] = root_file;
  detail::relay_blueprint_mesh_read (options, data);
  return bp2dray<Float> (data, import_order_policy);
}

} // namespace detail

DataSet BlueprintReader::load (const std::string &root_file, const ImportOrderPolicy &import_order_policy)
{
  return detail::load_bp (root_file, import_order_policy);
}

DataSet BlueprintReader::load (const std::string &root_file, const int cycle, const ImportOrderPolicy &import_order_policy)
{
  std::string full_root = detail::append_cycle (root_file, cycle) + ".root";
  return detail::load_bp (full_root, import_order_policy);
}

DataSet
BlueprintReader::blueprint_to_dray (const conduit::Node &n_dataset, const ImportOrderPolicy &import_order_policy)
{
  return detail::bp2dray<Float> (n_dataset, import_order_policy);
}

} // namespace dray
